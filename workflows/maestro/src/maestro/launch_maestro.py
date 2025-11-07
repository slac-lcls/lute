"""Launch a LUTE workflow using Maestro as the workflow manager."""

__all__ = ["load_lute_dag"]
__author__ = "Gabriel Dorlhiac"

from typing import List

from maestro._maestro import _maestro
from maestro.parser import load_lute_dag


def main():
    lute_location: str = "/home/dorlhiac/Descargas/lute_slurm"
    config_file: str = f"{lute_location}/config/test_local.yaml"
    debug: bool = True

    workflow_path: str = "/home/dorlhiac/Descargas/lute_slurm/test.dag"

    wf_defn: List[_maestro.JobStep] = load_lute_dag(
        workflow_path=workflow_path,
        lute_location=lute_location,
        config_file=config_file,
        debug=debug,
    )
    _maestro.run_workflow(wf_defn)


if __name__ == "__main__":
    main()
