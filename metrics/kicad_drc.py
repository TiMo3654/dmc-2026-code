import json
import os
import subprocess
import sys
import tempfile


def run_drc(pcb_path: str) -> dict:
    """
    Runs KiCad's Design Rule Check on a board using kicad-cli and returns
    the parsed JSON report (violations, unconnected items, etc. -- the
    exact schema depends on the installed KiCad version).

    Requires kicad-cli with "pcb drc" JSON output support (KiCad 8+).
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = tmp.name

    try:
        result = subprocess.run(
            [
                "kicad-cli", "pcb", "drc",
                pcb_path,
                "--output", report_path,
                "--format", "json",
                "--severity-all",
            ],
            capture_output=True, text=True,
        )
        # kicad-cli exits non-zero when violations are found -- that's an
        # expected outcome, not a failure. Only treat it as an error if
        # the report file was never written.
        if not os.path.exists(report_path) or os.path.getsize(report_path) == 0:
            raise RuntimeError(
                "kicad-cli DRC failed to produce a report.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        with open(report_path, "r") as f:
            report = json.load(f)
    finally:
        if os.path.exists(report_path):
            os.remove(report_path)

    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python kicad_drc.py <path-to-file.kicad_pcb>")
        sys.exit(1)

    report = run_drc(sys.argv[1])
    print(json.dumps(report))