import argparse
import json

import numpy as np


def relative_placement_metrics(ideal_positions, run_results, ic_designator):
    ic_ideal = np.array(ideal_positions[ic_designator])
    ic_runs = np.array(run_results[ic_designator])

    results = {}

    for comp, ideal_xy in ideal_positions.items():
        if comp == ic_designator or comp not in run_results:
            continue

        expert_vector = np.array(ideal_xy) - ic_ideal
        norm_expert = np.linalg.norm(expert_vector)

        component_runs = np.array(run_results[comp])
        run_vectors = component_runs - ic_runs
        norm_runs = np.linalg.norm(run_vectors, axis=1)

        if norm_expert < 1e-9:
            angle_error_deg = np.full(len(run_vectors), np.nan)
        else:
            cross = (
                expert_vector[0] * run_vectors[:, 1]
                - expert_vector[1] * run_vectors[:, 0]
            )
            dot = run_vectors @ expert_vector
            angle_error_deg = np.degrees(np.arctan2(cross, dot))
            angle_error_deg[norm_runs < 1e-9] = np.nan

        distance_error = norm_runs - norm_expert
        vector_error = np.linalg.norm(run_vectors - expert_vector, axis=1)

        results[comp] = {
            "angle_error_deg": angle_error_deg,
            "distance_error": distance_error,
            "vector_error": vector_error,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate placement metrics relative to an IC designator."
    )
    parser.add_argument("ideal_positions", help="JSON file of {ref: [x, y]}")
    parser.add_argument("run_results", help="JSON file of {ref: [[x, y], ...]}")
    parser.add_argument("ic_designator", help="Reference designator for the IC")
    args = parser.parse_args()

    with open(args.ideal_positions, "r", encoding="utf-8") as file:
        ideal_positions = json.load(file)
    with open(args.run_results, "r", encoding="utf-8") as file:
        run_results = json.load(file)

    metrics = relative_placement_metrics(
        ideal_positions,
        run_results,
        args.ic_designator,
    )

    serializable_metrics = {
        comp: {
            name: values.tolist()
            for name, values in component_metrics.items()
        }
        for comp, component_metrics in metrics.items()
    }
    print(json.dumps(serializable_metrics, indent=2))


if __name__ == "__main__":
    main()
