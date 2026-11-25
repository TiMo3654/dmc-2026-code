"""
kicad_placement.py

A placement-optimization interface for KiCad boards, built on pcbnew.

Exposes a `PlacementProblem` class that any optimizer (simulated annealing,
genetic algorithm, etc.) can drive:

  - `problem.state`            current placement (dict of ref -> ComponentState)
  - `problem.evaluate(state)`  cost = weighted(area, HPWL) + penalties
  - `problem.random_neighbor(state)`  small random move/rotation of one part
  - `problem.random_state()`   fully random placement (for GA population init)
  - `problem.save(path, state)`  commit a state back into the board and write it out

Design notes (see chat for rationale):
  - Rotation is restricted to {0, 90, 180, 270} degrees.
  - Component footprint geometry for area/overlap/out-of-bounds purposes is
    the exact axis-aligned bounding box of the footprint's pads and
    non-text graphic items (silkscreen, fab, courtyard, ...), stored as
    offsets relative to the anchor -- NOT assumed to be centered on the
    anchor, since library footprints are frequently anchored off-center
    (e.g. at pin 1). Reference/value text is excluded from this bbox, so
    it does not affect placement cost; pad positions used for HPWL are
    exact regardless.
  - Board edge is read from the Edge.Cuts layer unless overridden.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pcbnew

ALLOWED_ROTATIONS: Tuple[int, ...] = (0, 90, 180, 270)


def _make_point(x_nm: float, y_nm: float):
    """Construct a pcbnew point, compatible with both VECTOR2I (KiCad 7+)
    and wxPoint (older) APIs."""
    try:
        return pcbnew.VECTOR2I(int(x_nm), int(y_nm))
    except AttributeError:
        return pcbnew.wxPoint(int(x_nm), int(y_nm))


def _rect_overlap_area(a: Tuple[float, float, float, float],
                        b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return ox * oy


def _out_of_bounds_area(rect: Tuple[float, float, float, float],
                         board_bbox: Tuple[float, float, float, float]) -> float:
    rx0, ry0, rx1, ry1 = rect
    total_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    inside_area = _rect_overlap_area(rect, board_bbox)
    return max(0.0, total_area - inside_area)


_TEXT_CLASSES = {"FP_TEXT", "PCB_TEXT", "PCB_TEXTBOX", "FP_TEXTBOX"}


def _bbox_from_items(items) -> Optional[Tuple[float, float, float, float]]:
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    found = False
    for item in items:
        found = True
        bbox = item.GetBoundingBox()
        min_x = min(min_x, bbox.GetLeft())
        min_y = min(min_y, bbox.GetTop())
        max_x = max(max_x, bbox.GetRight())
        max_y = max(max_y, bbox.GetBottom())
    return (min_x, min_y, max_x, max_y) if found else None


def _footprint_geom_bbox_nm(fp) -> Tuple[Tuple[float, float, float, float], bool]:
    """
    Bounding box (nanometers, board coordinates) representing a footprint's
    placement extent. Returns (bbox, used_courtyard).

    Prefers the footprint's courtyard graphics (F.CrtYd / B.CrtYd) -- the
    layer KiCad itself uses for inter-footprint clearance/overlap checks.
    Falls back to pads + non-text graphic items (silkscreen, fab, ...) for
    footprints that don't define a courtyard, and finally to the anchor
    point alone if the footprint has neither.
    """
    graphical_items = list(fp.GraphicalItems()) if hasattr(fp, "GraphicalItems") else []

    courtyard_items = [
        item for item in graphical_items
        if item.GetLayer() in (pcbnew.F_CrtYd, pcbnew.B_CrtYd)
    ]
    bbox = _bbox_from_items(courtyard_items)
    if bbox is not None:
        return bbox, True

    fallback_items = [pad for pad in fp.Pads()] + [
        item for item in graphical_items if item.GetClass() not in _TEXT_CLASSES
    ]
    bbox = _bbox_from_items(fallback_items)
    if bbox is not None:
        return bbox, False

    pos = fp.GetPosition()
    return (float(pos.x), float(pos.y), float(pos.x), float(pos.y)), False


@dataclass(frozen=True)
class ComponentState:
    ref: str
    x: float          # mm, board coordinates
    y: float          # mm, board coordinates
    rotation: int      # degrees, one of ALLOWED_ROTATIONS


PlacementState = Dict[str, ComponentState]


class PlacementProblem:
    def __init__(
        self,
        pcb_path: str,
        movable_refs: Optional[List[str]] = None,
        fixed_refs: Optional[List[str]] = None,
        board_bbox: Optional[Tuple[float, float, float, float]] = None,
        weight_area: float = 1.0,
        weight_hpwl: float = 1.0,
        penalty_overlap: float = 100.0,
        penalty_out_of_bounds: float = 100.0,
        allowed_rotations: Tuple[int, ...] = ALLOWED_ROTATIONS,
    ):
        """
        movable_refs / fixed_refs: at most one should be given.
          - movable_refs: only these refs are optimized; everything else is
            fixed in place (but still counts toward area/HPWL/overlap/OOB).
          - fixed_refs: these refs are fixed in place; every other
            footprint on the board is movable. Convenient when only a
            handful of parts (e.g. connectors) need to stay put.
          - neither given: every footprint on the board is movable.
        """
        self.pcb_path = pcb_path
        self.board = pcbnew.LoadBoard(pcb_path)
        self.to_mm = pcbnew.ToMM
        self.from_mm = pcbnew.FromMM
        self.allowed_rotations = allowed_rotations

        self._fps = {fp.GetReference(): fp for fp in self.board.GetFootprints()}
        self.refs: List[str] = list(self._fps.keys())

        if movable_refs is not None and fixed_refs is not None:
            raise ValueError("Pass at most one of movable_refs / fixed_refs, not both.")

        if fixed_refs is not None:
            unknown = set(fixed_refs) - set(self.refs)
            if unknown:
                raise ValueError(f"fixed_refs not found on board: {sorted(unknown)}")
            self.movable_refs: List[str] = [r for r in self.refs if r not in set(fixed_refs)]
        elif movable_refs is not None:
            unknown = set(movable_refs) - set(self.refs)
            if unknown:
                raise ValueError(f"movable_refs not found on board: {sorted(unknown)}")
            self.movable_refs = list(movable_refs)
        else:
            self.movable_refs = list(self.refs)

        self.fixed_refs: List[str] = [r for r in self.refs if r not in set(self.movable_refs)]

        self.weight_area = weight_area
        self.weight_hpwl = weight_hpwl
        self.penalty_overlap = penalty_overlap
        self.penalty_out_of_bounds = penalty_out_of_bounds

        self.board_bbox = board_bbox or self._compute_edge_cuts_bbox()

        # Precompute, per footprint per allowed rotation: bbox size (mm)
        # and pad (net, local_dx, local_dy) offsets relative to the
        # footprint anchor, in mm. Done once; requires temporarily
        # rotating the in-memory footprints (not saved to disk here).
        self._geometry = self._precompute_geometry()

        # Initial state = current placement on the loaded board, with
        # rotation snapped to the nearest allowed angle.
        self.state: PlacementState = {}
        for ref, fp in self._fps.items():
            pos = fp.GetPosition()
            self.state[ref] = ComponentState(
                ref=ref,
                x=self.to_mm(pos.x),
                y=self.to_mm(pos.y),
                rotation=self._snap_rotation(fp.GetOrientationDegrees()),
            )

    # -- setup -------------------------------------------------------

    def _snap_rotation(self, angle_deg: float) -> int:
        angle_deg = angle_deg % 360
        return min(self.allowed_rotations, key=lambda a: abs(a - angle_deg))

    def _compute_edge_cuts_bbox(self) -> Tuple[float, float, float, float]:
        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        for drawing in self.board.GetDrawings():
            if drawing.GetLayer() != pcbnew.Edge_Cuts:
                continue
            bbox = drawing.GetBoundingBox()
            min_x = min(min_x, self.to_mm(bbox.GetLeft()))
            min_y = min(min_y, self.to_mm(bbox.GetTop()))
            max_x = max(max_x, self.to_mm(bbox.GetRight()))
            max_y = max(max_y, self.to_mm(bbox.GetBottom()))
        if min_x == math.inf:
            raise ValueError(
                "No Edge.Cuts geometry found on board; pass board_bbox=(min_x, "
                "min_y, max_x, max_y) explicitly (in mm)."
            )
        return (min_x, min_y, max_x, max_y)

    def _precompute_geometry(self) -> dict:
        geometry = {}
        no_courtyard_refs = []
        for ref, fp in self._fps.items():
            orig_angle = fp.GetOrientationDegrees()
            anchor = fp.GetPosition()
            per_angle = {}
            used_courtyard_any_angle = True
            for angle in self.allowed_rotations:
                fp.SetOrientationDegrees(angle)

                (bx0, by0, bx1, by1), used_courtyard = _footprint_geom_bbox_nm(fp)
                used_courtyard_any_angle = used_courtyard_any_angle and used_courtyard
                rect_offset = (
                    self.to_mm(bx0 - anchor.x),
                    self.to_mm(by0 - anchor.y),
                    self.to_mm(bx1 - anchor.x),
                    self.to_mm(by1 - anchor.y),
                )

                pads = []
                for pad in fp.Pads():
                    net_name = pad.GetNetname()
                    if net_name == "":
                        continue
                    ppos = pad.GetPosition()
                    dx = self.to_mm(ppos.x - anchor.x)
                    dy = self.to_mm(ppos.y - anchor.y)
                    pads.append((net_name, dx, dy))

                per_angle[angle] = {"rect_offset": rect_offset, "pads": pads}
            fp.SetOrientationDegrees(orig_angle)  # restore original pose
            geometry[ref] = per_angle
            if not used_courtyard_any_angle:
                no_courtyard_refs.append(ref)

        if no_courtyard_refs:
            print(
                "[kicad_placement] Warning: no courtyard graphics found for "
                f"{len(no_courtyard_refs)} footprint(s), falling back to "
                f"pads+silkscreen/fab bbox: {', '.join(sorted(no_courtyard_refs))}"
            )

        return geometry

    # -- state utilities ----------------------------------------------

    def random_state(self, margin: float = 2.0) -> PlacementState:
        """A fully random placement, useful for GA population init."""
        bx0, by0, bx1, by1 = self.board_bbox
        new_state = dict(self.state)
        for ref in self.movable_refs:
            new_state[ref] = ComponentState(
                ref=ref,
                x=random.uniform(bx0 + margin, bx1 - margin),
                y=random.uniform(by0 + margin, by1 - margin),
                rotation=random.choice(self.allowed_rotations),
            )
        return new_state

    def random_neighbor(
        self,
        state: PlacementState,
        move_std: float = 5.0,
        rotate_prob: float = 0.2,
    ) -> PlacementState:
        """Perturb one random movable component: either a small Gaussian
        move, or a rotation to a (different) allowed angle."""
        ref = random.choice(self.movable_refs)
        s = state[ref]
        new_state = dict(state)
        if random.random() < rotate_prob:
            choices = [a for a in self.allowed_rotations if a != s.rotation]
            new_rotation = random.choice(choices) if choices else s.rotation
            new_state[ref] = ComponentState(ref, s.x, s.y, new_rotation)
        else:
            new_x = s.x + random.gauss(0, move_std)
            new_y = s.y + random.gauss(0, move_std)
            new_state[ref] = ComponentState(ref, new_x, new_y, s.rotation)
        return new_state

    # -- evaluation -----------------------------------------------------

    def _world_rects(self, state: PlacementState) -> Dict[str, Tuple[float, float, float, float]]:
        rects = {}
        for ref in self.refs:
            s = state.get(ref, self.state[ref])
            dx0, dy0, dx1, dy1 = self._geometry[ref][s.rotation]["rect_offset"]
            rects[ref] = (s.x + dx0, s.y + dy0, s.x + dx1, s.y + dy1)
        return rects

    def evaluate(self, state: Optional[PlacementState] = None) -> dict:
        state = state if state is not None else self.state
        rects = self._world_rects(state)

        # 1. overall placement area (bbox enclosing every component)
        xs0 = min(r[0] for r in rects.values())
        ys0 = min(r[1] for r in rects.values())
        xs1 = max(r[2] for r in rects.values())
        ys1 = max(r[3] for r in rects.values())
        area = (xs1 - xs0) * (ys1 - ys0)

        # 2. HPWL over all nets, using every component's pads (fixed + movable)
        net_points = defaultdict(list)
        for ref in self.refs:
            s = state.get(ref, self.state[ref])
            for net_name, dx, dy in self._geometry[ref][s.rotation]["pads"]:
                net_points[net_name].append((s.x + dx, s.y + dy))

        hpwl_total = 0.0
        for pts in net_points.values():
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            hpwl_total += (max(xs) - min(xs)) + (max(ys) - min(ys))

        # 3. overlap penalty: sum of pairwise overlap areas involving at
        #    least one movable component (fixed-fixed pairs assumed valid)
        overlap_area = 0.0
        for i in range(len(self.refs)):
            for j in range(i + 1, len(self.refs)):
                r1, r2 = self.refs[i], self.refs[j]
                if r1 not in self.movable_refs and r2 not in self.movable_refs:
                    continue
                overlap_area += _rect_overlap_area(rects[r1], rects[r2])

        # 4. out-of-bounds penalty: movable components only
        oob_area = 0.0
        for ref in self.movable_refs:
            oob_area += _out_of_bounds_area(rects[ref], self.board_bbox)

        cost = (
            self.weight_area * area
            + self.weight_hpwl * hpwl_total
            + self.penalty_overlap * overlap_area
            + self.penalty_out_of_bounds * oob_area
        )

        return {
            "cost": cost,
            "area": area,
            "hpwl": hpwl_total,
            "overlap_area": overlap_area,
            "oob_area": oob_area,
        }

    # -- committing results --------------------------------------------

    def apply_state(self, state: PlacementState) -> None:
        """Write a state into the live pcbnew board object (in memory)."""
        for ref, s in state.items():
            fp = self._fps[ref]
            fp.SetPosition(_make_point(self.from_mm(s.x), self.from_mm(s.y)))
            fp.SetOrientationDegrees(s.rotation)

    def save(self, out_path: str, state: Optional[PlacementState] = None) -> None:
        self.apply_state(state if state is not None else self.state)
        self.board.Save(out_path)