import argparse

from dynamic import dynamic_flow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="create_precect_deployment",
        description="Create Prefect deployment for dynamic_flow.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Name of the new deployment.",
        type=str,
        default="dev",
    )

    args: argparse.Namespace = parser.parse_args()
    dynamic_flow.from_source(
        source="https://github.com/slac-lcls/lute.git",
        entrypoint="workflows/prefect/dynamic.py:dynamic_flow",
    ).deploy(
        name=args.name,
        work_pool_name="psdm-prefect-workers",
    )
