import pcbnew
from typing import Dict, List, Tuple


def get_component_positions(pcb_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns each footprint's reference designator mapped to a list
    containing its (x, y) position in mm, e.g.:

        {"R1": [(30.0, 45.0)], "R2": [(55.0, 40.0)], "C1": [(40.0, 20.0)]}

    The list-of-one-tuple shape matches the positions_by_ref format used
    by the placement-consistency plotting functions (which use lists to
    hold one point per optimization run); here there's just a single run
    (the board as loaded).
    """
    board = pcbnew.LoadBoard(pcb_path)
    to_mm = pcbnew.ToMM

    positions = {}
    for fp in board.GetFootprints():
        pos = fp.GetPosition()
        positions[fp.GetReference()] = [(to_mm(pos.x), to_mm(pos.y))]

    return positions


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python kicad_component_positions.py <path-to-file.kicad_pcb>")
        sys.exit(1)

    result = get_component_positions(sys.argv[1])
    print(json.dumps(result, indent=2))