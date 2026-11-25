import pcbnew


def get_pcb_bounding_box(pcb_path: str) -> dict | None:
    """
    Returns the bounding box (in mm) enclosing all placed footprints
    on a KiCad PCB, or None if the board has no footprints.
    """
    board = pcbnew.LoadBoard(pcb_path)
    footprints = board.GetFootprints()
    if not footprints:
        return None

    box = footprints[0].GetBoundingBox()
    for fp in footprints[1:]:
        box.Merge(fp.GetBoundingBox())

    to_mm = pcbnew.ToMM
    min_x, min_y = to_mm(box.GetLeft()), to_mm(box.GetTop())
    max_x, max_y = to_mm(box.GetRight()), to_mm(box.GetBottom())

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


if __name__ == "__main__":
    import sys
    import json

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    json_mode = "--json" in sys.argv

    if len(args) != 1:
        print("Usage: python kicad_bbox.py <path-to-file.kicad_pcb> [--json]")
        sys.exit(1)

    result = get_pcb_bounding_box(args[0])

    if json_mode:
        print(json.dumps(result))
    elif result is None:
        print("No footprints found on this board.")
    else:
        print(f"Bounding box (mm): {result}")
        print(f"  Width:  {result['width']:.3f} mm")
        print(f"  Height: {result['height']:.3f} mm")