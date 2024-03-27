"""Test Airflow <-> JID requests.

Minimal example to test requests from Airflow to the JID.
"""

from datetime import datetime
import os
from airflow import DAG
from lute.operators.jidoperators import RequestOnlyOperator

dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"
description: str = "Test Airflow <-> JID API"

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(1970, 1, 1),
    schedule_interval=None,
    description=description,
)

requester: RequestOnlyOperator = RequestOnlyOperator(task_id="MakeRequest", dag=dag)

requester
