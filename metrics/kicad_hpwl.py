import pcbnew
from collections import defaultdict


def get_net_hpwl(pcb_path: str, exclude_no_net: bool = True) -> dict:
    """
    Computes the half-perimeter wirelength (HPWL) per net on a KiCad PCB,
    based on pad (pin) positions.

    HPWL of a net = width + height of the bounding box enclosing all of
    that net's pads (i.e. half the perimeter of the bounding box).
    Nets with 0 or 1 pads have HPWL 0.

    Args:
        pcb_path: path to a .kicad_pcb file.
        exclude_no_net: if True (default), pads with no net assigned
            (unrouted pads, net name "") are ignored.

    Returns:
        dict with:
            "per_net": {net_name: hpwl_mm, ...}
            "total": sum of all per-net HPWLs (mm)
            "num_nets": number of nets considered
    """
    board = pcbnew.LoadBoard(pcb_path)
    to_mm = pcbnew.ToMM

    net_pads = defaultdict(list)
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            net_name = pad.GetNetname()
            if exclude_no_net and net_name == "":
                continue
            pos = pad.GetPosition()
            net_pads[net_name].append((to_mm(pos.x), to_mm(pos.y)))

    per_net_hpwl = {}
    for net_name, coords in net_pads.items():
        if len(coords) < 2:
            per_net_hpwl[net_name] = 0.0
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        per_net_hpwl[net_name] = width + height

    total = sum(per_net_hpwl.values())

    return {
        "per_net": per_net_hpwl,
        "total": total,
        "num_nets": len(per_net_hpwl),
    }


if __name__ == "__main__":
    import sys
    import json

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    json_mode = "--json" in sys.argv

    if len(args) != 1:
        print("Usage: python kicad_hpwl.py <path-to-file.kicad_pcb> [--json]")
        sys.exit(1)

    result = get_net_hpwl(args[0])

    if json_mode:
        print(json.dumps(result))
    else:
        print(f"Nets considered: {result['num_nets']}")
        print(f"Total HPWL: {result['total']:.3f} mm\n")
        print("Per-net HPWL (top 10 by length):")
        top = sorted(result["per_net"].items(), key=lambda kv: kv[1], reverse=True)[:10]
        for net_name, hpwl in top:
            print(f"  {net_name:30s} {hpwl:.3f} mm")