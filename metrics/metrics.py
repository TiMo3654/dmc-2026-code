import subprocess
import json
import os

def get_pcb_bbox_via_podman(pcb_relative_path: str) -> dict:
    """
    pcb_relative_path is relative to ~/kicad_workspace, matching how
    it's mounted into the container.
    """
    result = subprocess.run(
        [
            "podman", "run", "--rm",
            "-v", f"{os.environ['HOME']}/kicad_workspace:/workspace/projects:Z",
            "kicad-mcp",
            "python3", "/workspace/projects/kicad_bbox.py",
            f"/workspace/projects/{pcb_relative_path}",
            "--json",
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)



def get_hpwl_via_podman(pcb_relative_path: str) -> dict:
    result = subprocess.run(
        [
            "podman", "run", "--rm",
            "-v", f"{os.environ['HOME']}/kicad_workspace:/workspace/projects:Z",
            "kicad-mcp",
            "python3", "/workspace/projects/kicad_hpwl.py",
            f"/workspace/projects/{pcb_relative_path}",
            "--json",
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def get_component_positions_via_podman(pcb_relative_path: str) -> dict:
    result = subprocess.run(
        [
            "podman", "run", "--rm",
            "-v", f"{os.environ['HOME']}/kicad_workspace:/workspace/projects:Z",
            "kicad-mcp",
            "python3", "/workspace/projects/kicad_component_positions.py",
            f"/workspace/projects/{pcb_relative_path}",
        ],
        capture_output=True, text=True, check=True,
    )

    raw = json.loads(result.stdout)

    return {ref: [tuple(point) for point in points] for ref, points in raw.items()}


def run_drc_via_podman(pcb_relative_path: str) -> dict:
    result = subprocess.run(
        [
            "podman", "run", "--rm",
            "-v", f"{os.environ['HOME']}/kicad_workspace:/workspace/projects:Z",
            "kicad-mcp",
            "python3", "/workspace/projects/kicad_drc.py",
            f"/workspace/projects/{pcb_relative_path}",
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def set_min_clearance_via_podman(pcb_relative_path: str, clearance_mm: float, output_relative_path: str = None) -> None:
    args = [pcb_relative_path, str(clearance_mm)]
    if output_relative_path:
        args.append(output_relative_path)

    subprocess.run(
        [
            "podman", "run", "--rm",
            "-v", f"{os.environ['HOME']}/kicad_workspace:/workspace/projects:Z",
            "kicad-mcp",
            "python3", "/workspace/projects/kicad_design_rule.py",
            *[f"/workspace/projects/{a}" if not a.replace(".", "").isdigit() else a for a in args],
        ],
        check=True,
    )



if __name__ == "__main__":

    from metrics.vector_similarity import relative_placement_metrics

    print(f" Violations before Change: {len(run_drc_via_podman("test_board/test_board_opt_SA.kicad_pcb")["violations"])}")   

    set_min_clearance_via_podman(pcb_relative_path="test_board/test_board_opt_SA.kicad_pcb", clearance_mm=0.15)

    print(f"Violations after Change: {len(run_drc_via_podman("test_board/test_board_opt_SA.kicad_pcb")["violations"])}") 



    # result_bbox = get_pcb_bbox_via_podman("buck_converter_dmc/test_board.kicad_pcb")

    # result_hpwl = get_hpwl_via_podman("buck_converter_dmc/test_board.kicad_pcb")

    # result_positions = get_component_positions_via_podman("buck_converter_dmc/test_board.kicad_pcb")


    # ideal_positions = {
    #     "R1": (141.0, 100.0),
    #     "R2": (137.5, 100.0),
    #     "C1": (133.5, 100.0),
    #     "C2": (135.5, 106.5),
    #     "C3": (138.0, 104.0),
    #     "U1": (135.2, 102.5),
    #     "L1": (130.5, 105.5),
    # }

    # print(relative_placement_metrics(ideal_positions=ideal_positions, run_results=result_positions, ic_designator="U1"))

    # print(result_bbox)
    # print(result_hpwl)
    # print(result_positions)