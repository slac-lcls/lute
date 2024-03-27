"""SFX Processing DAG (W/ Experimental Phasing)

Runs SFX processing, beginning with the PyAlgos peak finding algorithm.
This workflow is used for data requiring experimental phasing. Do NOT use this
DAG if you are using molecular replacement.

Note:
    The task_id MUST match the managed task name when defining DAGs - it is used
    by the operator to properly launch it.

    dag_id names must be unique, and they are not namespaced via folder
    hierarchy. I.e. all DAGs on an Airflow instance must have unique ids. The
    Airflow instance used by LUTE is currently shared by other software - DAG
    IDs should always be prefixed with `lute_`. LUTE scripts should append this
    internally, so a DAG "lute_test" can be triggered by asking for "test"
"""

from datetime import datetime
import os
from airflow import DAG
from lute.operators.jidoperators import JIDSlurmOperator

dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"
description: str = (
    "Run SFX processing using PyAlgos peak finding and experimental phasing"
)

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(2024, 3, 18),
    schedule_interval=None,
    description=description,
)

peak_finder: JIDSlurmOperator = JIDSlurmOperator(task_id="PeakFinderPyAlgos", dag=dag)

indexer: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=120, task_id="CrystFELIndexer", dag=dag
)

# Merge
merger: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=120, task_id="PartialatorMerger", dag=dag
)

# Figures of merit
hkl_comparer: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=8, task_id="HKLComparer", dag=dag
)

# HKL conversions
hkl_manipulator: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=8, task_id="HKLManipulator", dag=dag
)

# SHELX Tasks
shelxc: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=20, task_id="SHELXCRunner", dag=dag
)


peak_finder >> indexer >> merger >> hkl_manipulator >> shelxc
merger >> hkl_comparer

# Run summaries
