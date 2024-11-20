"""
setup.py for LUTE
"""

from setuptools import find_packages, setup

USE_MYPYC: bool
try:
    from mypyc.build import mypycify

    USE_MYPYC = True
except ModuleNotFoundError:
    USE_MYPYC = False

version_fh = open("lute/__init__.py", "r")
version = version_fh.readlines()[-1].split("=")[1].strip().split('"')[1]
version_fh.close()
setup(
    name="lute",
    version=version,
    url="https://slac-lcls.github.io/lute/dev/",
    description="LCLS Unified Task Executor.",
    install_requires=[
        "numpy",
        "pydantic==1.10.13",
        "requests",
        "zmq",
        "jinja2",
        # "mpi4py",
    ],
    extras_require={
        "docs": [
            "mkdocs",
            "mkdocstring",
            "mkdocstring-python",
            "mkdocs-click",
            "mkdocs-material",
            "mkdocs-material-extensions",
        ],
    },
    packages=find_packages(where="."),
    ext_modules=(mypycify(["lute/execution", "lute/tasks"]) if USE_MYPYC else []),
    package_data={"config": ["*.yaml", "templates/*.*"]},
    platforms="any",
    scripts=[
        "run_task.py",
        "subprocess_task.py",
        "launch_scripts/launch_airflow.py",
        "launch_scripts/submit_slurm.sh",
        "launch_scripts/submit_launch_airflow.sh",
        "utilities/activate_installation",
    ],
)
