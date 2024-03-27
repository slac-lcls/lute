"""Airflow Operators for running LUTE tasks via Airflow.

Operators submit managed tasks to run and monitor task status. Status is
reported to Airflow which manages the execution order of a directed acyclic
graph (DAG) to determine which managed task to submit and when.

Classes:
    JIDSlurmOperator: Submits a managed task to run on S3DF batch nodes via the
        job interface daemon (JID). Airflow itself has no access to data or the
        file system mounted on the batch node so submission and monitoring is
        done exclusively via the JID API.
"""

__all__ = ["JIDSlurmOperator", "RequestOnlyOperator"]
__author__ = "Fred Poitevin, Murali Shankar"

import uuid
import getpass
import time
import logging
import re
from typing import Dict, Any, Union, List, Optional

import requests

from airflow.models import BaseOperator
from airflow.exceptions import AirflowException
from airflow.plugins_manager import AirflowPlugin
from airflow.utils.decorators import apply_defaults

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


class RequestOnlyOperator(BaseOperator):
    """This Operator makes a JID request and exits."""

    @apply_defaults
    def __init__(
        self,
        user: str = getpass.getuser(),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # Initializes self.task_id
        self.user: str = user

    def execute(self, context: Dict[str, Any]) -> None:
        """Method called by Airflow which submits SLURM Job via JID.

        Args:
            context (Dict[str, Any]): Airflow dictionary object.
                https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html
                contains a list of available variables and their description.
        """
        # logger.info(f"Attempting to run at {self.get_location(context)}...")
        logger.info(f"Attempting to run at S3DF.")
        dagrun_config: Dict[str, Union[str, Dict[str, Union[str, int, List[str]]]]] = (
            context.get("dag_run").conf
        )
        jid_job_definition: Dict[str, str] = {
            "_id": str(uuid.uuid4()),
            "name": self.task_id,
            "executable": f"myexecutable.sh",
            "trigger": "MANUAL",
            "location": dagrun_config.get("ARP_LOCATION", "S3DF"),
            "parameters": "--partition=milano --account=lcls:data",
            "run_as_user": self.user,
        }

        control_doc: Dict[str, Union[str, Dict[str, str]]] = {
            "_id": str(uuid.uuid4()),
            "arp_root_job_id": dagrun_config.get("ARP_ROOT_JOB_ID"),
            "experiment": dagrun_config.get("experiment"),
            "run_num": dagrun_config.get("run_id"),
            "user": dagrun_config.get("user"),
            "status": "",
            "tool_id": "",
            "def_id": str(uuid.uuid4()),
            "def": jid_job_definition,
        }

        uri: str = (
            "https://psdm.slac.stanford.edu/arps3dfjid/jid/ws/{experiment}/start_job"
        )
        # Endpoints have the string "{experiment}" in them
        uri = uri.format(experiment=dagrun_config.get("experiment"))
        auth: Any = dagrun_config.get("Authorization")
        logger.info(f"Calling {uri} with {control_doc}...")

        logger.info(requests.__file__)
        resp: requests.models.Response = requests.post(
            uri, json=control_doc, headers={"Authorization": auth}
        )
        logger.info(f"Status code: {resp.status_code}")
        logger.info(requests.__file__)


class JIDSlurmOperator(BaseOperator):
    """Airflow Operator which submits SLURM jobs through the JID."""

    ui_color: str = "#006699"

    jid_api_location: str = "https://psdm.slac.stanford.edu/arps3dfjid/jid/ws"
    """S3DF JID API location."""

    jid_api_endpoints: Dict[str, str] = {
        "start_job": "{experiment}/start_job",
        "job_statuses": "job_statuses",
        "job_log_file": "{experiment}/job_log_file",
    }

    @apply_defaults
    def __init__(
        self,
        user: str = getpass.getuser(),
        poke_interval: float = 30.0,
        max_cores: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # Initializes self.task_id
        self.lute_location: str = ""
        self.user: str = user
        self.poke_interval: float = poke_interval
        self.max_cores: Optional[int] = max_cores

    def create_control_doc(
        self, context: Dict[str, Any]
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """Prepare the control document for job submission via the JID.

        Translates and Airflow dictionary to the representation needed by the
        JID.

        Args:
            context (Dict[str, Any]): Airflow dictionary object.
                https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html
                contains a list of available variables and their description.

        Returns:
            control_doc (Dict[str, Union[str, Dict[str, str]]]): JID job control
                dictionary.
        """

        dagrun_config: Dict[str, Union[str, Dict[str, Union[str, int, List[str]]]]] = (
            context.get("dag_run").conf
        )

        self.lute_location = dagrun_config.get(
            "lute_location", "/sdf/group/lcls/ds/tools/lute/latest"
        )
        lute_params: Dict[str, str] = dagrun_config.get("lute_params", {})

        config_path: str = lute_params["config_file"]
        # Note that task_id is from the parent class.
        # When defining the Operator instances the id is assumed to match a
        # managed task!
        lute_param_str: str
        if lute_params["debug"]:
            lute_param_str = f"--taskname {self.task_id} --config {config_path} --debug"
        else:
            lute_param_str = f"--taskname {self.task_id} --config {config_path}"

        # slurm_params holds a List[str]
        slurm_param_str: str = " ".join(dagrun_config.get("slurm_params"))
        # Cap max cores used by a managed Task if that is requested
        pattern: str = r"(?<=\bntasks=)\d+"
        ntasks: int
        try:
            ntasks = int(re.findall(pattern, slurm_param_str)[0])
        except IndexError as err:  # If `ntasks` not passed - 1 is default
            ntasks = 1
        if self.max_cores is not None and ntasks > self.max_cores:
            slurm_param_str = re.sub(pattern, f"{self.max_cores}", slurm_param_str)

        parameter_str: str = f"{lute_param_str} {slurm_param_str}"

        jid_job_definition: Dict[str, str] = {
            "_id": str(uuid.uuid4()),
            "name": self.task_id,
            "executable": f"{self.lute_location}/launch_scripts/submit_slurm.sh",
            "trigger": "MANUAL",
            "location": dagrun_config.get("ARP_LOCATION", "S3DF"),
            "parameters": parameter_str,
            "run_as_user": self.user,
        }

        control_doc: Dict[str, Union[str, Dict[str, str]]] = {
            "_id": str(uuid.uuid4()),
            "arp_root_job_id": dagrun_config.get("ARP_ROOT_JOB_ID"),
            "experiment": dagrun_config.get("experiment"),
            "run_num": dagrun_config.get("run_id"),
            "user": dagrun_config.get("user"),
            "status": "",
            "tool_id": "",
            "def_id": str(uuid.uuid4()),
            "def": jid_job_definition,
        }

        return control_doc

    def parse_response(
        self, resp: requests.models.Response, check_for_error: List[str]
    ) -> Dict[str, Any]:
        """Parse a JID HTTP response.

        Args:
            resp (requests.models.Response): The response object from a JID
                HTTP request.
            check_for_error (List[str]): A list of strings/patterns to search
                for in response. Exception is raised if there are any matches.

        Returns:
            value (Dict[str, Any]): Dictionary containing HTTP response value.

        Raises:
            AirflowException: Raised to translate multiple errors into object
                properly handled by the Airflow server.
        """
        logger.info(f"{resp.status_code}: {resp.content}")
        if not resp.status_code in (200,):
            raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
        try:
            json: Dict[str, Union[str, int]] = resp.json()
            if not json.get("success", "") in (True,):
                raise AirflowException(f"Error from JID {resp}: {resp.content}")
            value: Dict[str, Any] = json.get("value")

            for pattern in check_for_error:
                if pattern in value:
                    raise AirflowException(
                        f"Response failed due to string match {pattern} against response {value}"
                    )
            return value
        except Exception as err:
            raise AirflowException(
                f"Response from JID not parseable, unknown error: {err}"
            )

    def rpc(
        self,
        endpoint: str,
        control_doc: Union[
            List[Dict[str, Union[str, Dict[str, str]]]],
            Dict[str, Union[str, Dict[str, str]]],
        ],
        context: Dict[str, Any],
        check_for_error: List[str] = [],
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Submit job via JID and retrieve responses.

        Remote Procedure Call (RPC).

        Args:
            endpoint (str): Which API endpoint to use.

            control_doc (Dict[str, Union[str, Dict[str, str]]]): Dictionary for
                JID call.

            context (Dict[str, Any]): Airflow dictionary object.
                https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html
                contains a list of available variables and their description.

            check_for_error (List[str]): A list of keywords to search for in a
                response to indicate error conditions. Default [].

        Returns:
            value (Dict[str, Any]): Dictionary containing HTTP response value.
        """
        # if not self.get_location(context) in self.locations:
        #     raise AirflowException(f"JID location {self.get_location(context)} is not configured")
        dagrun_config: Dict[str, Union[str, Dict[str, Union[str, int, List[str]]]]] = (
            context.get("dag_run").conf
        )
        experiment: str = dagrun_config.get("experiment")
        auth: Any = dagrun_config.get("Authorization")

        uri: str = f"{self.jid_api_location}/{self.jid_api_endpoints[endpoint]}"
        # Endpoints have the string "{experiment}" in them
        uri = uri.format(experiment=experiment)

        logger.info(f"Calling {uri} with {control_doc}...")

        resp: requests.models.Response = requests.post(
            uri, json=control_doc, headers={"Authorization": auth}
        )
        logger.info(f" + {resp.status_code}: {resp.content.decode('utf-8')}")

        value: Dict[str, Any] = self.parse_response(resp, check_for_error)

        return value

    def execute(self, context: Dict[str, Any]) -> None:
        """Method called by Airflow which submits SLURM Job via JID.

        Args:
            context (Dict[str, Any]): Airflow dictionary object.
                https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html
                contains a list of available variables and their description.
        """
        # logger.info(f"Attempting to run at {self.get_location(context)}...")
        logger.info(f"Attempting to run at S3DF.")
        control_doc = self.create_control_doc(context)
        logger.info(control_doc)
        logger.info(f"{self.jid_api_location}/{self.jid_api_endpoints['start_job']}")
        # start_job requires a dictionary
        msg: Dict[str, Any] = self.rpc(
            endpoint="start_job", control_doc=control_doc, context=context
        )
        logger.info(f"JobID {msg['tool_id']} successfully submitted!")

        jobs: List[Dict[str, Any]] = [msg]
        time.sleep(10)  # Wait for job to queue.... FIXME
        logger.info("Checking for job completion.")
        while jobs[0].get("status") in ("RUNNING", "SUBMITTED"):
            jobs = self.rpc(
                endpoint="job_statuses",
                control_doc=jobs,  # job_statuses requires a list
                context=context,
                check_for_error=[" error: ", "Traceback"],
            )
            time.sleep(self.poke_interval)

        # Logs out to xcom
        out = self.rpc("job_log_file", jobs[0], context)
        context["task_instance"].xcom_push(key="log", value=out)


class JIDPlugins(AirflowPlugin):
    name = "jid_plugins"
    operators = [JIDSlurmOperator]
