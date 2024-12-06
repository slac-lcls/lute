"""Script to run functional tests of various LUTE workflows."""

__author__ = "Gabriel Dorlhiac"

import sys
import os
import uuid
import getpass
import datetime
import logging
import argparse
import subprocess
import shutil
import time
from typing import (
    Dict,
    Union,
    List,
    Optional,
    Any,
    cast,
    TypedDict,
    overload,
    Literal,
)

import yaml
import json
import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError

# Requests, urllib have lots of debug statements. Only set level for this logger
logger: logging.Logger = logging.getLogger("Launch_Func_Tests")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


class DagRunConf(TypedDict):
    experiment: str
    run_id: str
    JID_UPDATE_COUNTERS: Optional[str]
    ARP_ROOT_JOB_ID: str
    ARP_LOCATION: str
    Authorization: str
    user: str
    lute_location: str
    kerb_file: Optional[str]
    lute_params: Dict[str, Union[str, bool]]
    slurm_params: List[str]
    workflow: Dict[str, Any]


class DagRunData(TypedDict):
    dag_run_id: str
    conf: DagRunConf


def _retrieve_pw(instance: str = "prod", is_admin: bool = False) -> str:
    user_type: str
    if is_admin:
        logger.debug("Running as operator.")
        user_type = "admin"
    else:
        logger.debug("Running as user.")
        user_type = "user"
    path: str = "/sdf/group/lcls/ds/tools/lute/airflow_{instance}_{user_type}.txt"
    if instance == "prod" or instance == "test":
        path = path.format(instance=instance, user_type=user_type)
    else:
        raise ValueError('`instance` must be either "test" or "prod"!')
    with open(path, "r") as f:
        pw: str = f.readline().strip()
    return pw


def _request_arp_token(exp: str, lifetime: int = 300) -> str:
    """Request an ARP token via Kerberos endpoint.

    A token is required for job submission.

    Args:
        exp (str): The experiment to request the token for. All tokens are
            scoped to a single experiment.

        lifetime (int): The lifetime, in minutes, of the token. After the token
            expires, it can no longer be used for job submission. The maximum
            time you can request is 480 minutes (i.e. 8 hours). NOTE: since this
            token is used for the entirety of a workflow, it must have a lifetime
            equal or longer than the duration of the workflow's execution time.
    """
    from kerberos import GSSError  # type: ignore
    from krtc import KerberosTicket  # type: ignore

    try:
        krbheaders: Dict[str, str] = KerberosTicket(
            "HTTP@pswww.slac.stanford.edu"
        ).getAuthHeaders()
    except GSSError:
        logger.info(
            "Cannot proceed without credentials. Try running `kinit` from the command-line."
        )
        raise
    base_url: str = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk"
    token_endpoint: str = (
        f"{base_url}/{exp}/ws/generate_arp_token?token_lifetime={lifetime}"
    )
    resp: requests.models.Response = requests.get(token_endpoint, headers=krbheaders)
    resp.raise_for_status()
    token: str = resp.json()["value"]
    formatted_token: str = f"Bearer {token}"
    return formatted_token


def modify_permissions(path: str, permissions: int) -> None:
    """Recursively set permissions for a path."""
    os.chmod(path, permissions)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), permissions)

        for f in files:
            os.chmod(os.path.join(root, f), permissions)


@overload
def _run_subprocess_log(cmd: List[str], return_output: Literal[True]) -> str: ...


@overload
def _run_subprocess_log(cmd: List[str], return_output: Literal[False]) -> None: ...


@overload
def _run_subprocess_log(cmd: List[str]) -> None: ...


def _run_subprocess_log(cmd: List[str], return_output: bool = False) -> Optional[str]:
    """Run a subprocess with logging."""
    global logger

    out: str
    err: str
    out, err = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()
    if out:
        logger.info(out)
    if err:
        logger.info(err)

    if return_output:
        return out


def grep(match_str: str, in_file: str) -> List[str]:
    """Grep for specific lines of text output.

    Args:
        match_str (str): String to search for.

        in_file (str): File to search.

    Returns:
        lines (List[str]): The matches. It may be a list with just an empty
            string if nothing is found.
    """
    cmd: List[str] = ["grep", match_str, in_file]
    out: str = _run_subprocess_log(cmd, return_output=True)
    lines: List[str] = out.split("\n")
    return lines


def parse_yaml_value(input_config_yaml: str, key: str) -> str:
    """Retrieve a value for a specific YAML key.

    Currently only works for the header config section of a LUTE YAML file.
    E.g. for retrieving the experiment or run number.
    """
    lines: List[str] = grep(f"^{key}", input_config_yaml)
    key_val: str = lines[0]
    if "{{" in key_val:
        raise RuntimeError(
            "Substitutions not allowed in test configuration YAMLs! "
            f"Replace {key_val} with a defined value!"
        )
    val: str = key_val.split(":")[1].strip().replace('"', "")
    return val


def git_clone(repo: str, location: str, permissions: int) -> None:
    """Clone a git repository.

    Will not overwrite a directory of there is already a folder at the specified
    location.
    Args:
        repo (str): Name of the repository to clone. Should be specified as:
            "<user_or_organization>/<repository_name>"

        location (str): Path to the location to clone to.
    """
    global logger

    repo_only: str = repo.split("/")[1]
    if os.path.exists(f"{location}/{repo_only}"):
        logger.debug(
            f"Repository {repo} already exists at {location}. Will not overwrite."
        )
        return
    cmd: List[str] = [
        "git",
        "clone",
        f"https://github.com/{repo}.git",
        f"{location}/{repo_only}",
    ]
    _run_subprocess_log(cmd)
    modify_permissions(f"{location}/{repo_only}", permissions)


def inplace_sed(in_file: str, pattern: str) -> None:
    """Perform an in-place operation on a file using sed.

    Args:
        in_file (str): Path to the file to perform the substitution on.

        pattern (str): Operation. E.g. substitute with "s/old_text/new_text/g"
    """
    cmd: List[str] = ["sed", "-i", pattern, in_file]
    _run_subprocess_log(cmd)


def git_checkout_branch(path_to_repo: str, branch: str) -> None:
    old_cwd: str = os.getcwd()
    os.chdir(path_to_repo)
    switch_cmd: List[str] = ["git", "switch", branch]
    _run_subprocess_log(switch_cmd)
    os.chdir(old_cwd)


def git_fetch_pr_branch(path_to_repo: str, github_id: int) -> None:
    old_cwd: str = os.getcwd()
    os.chdir(path_to_repo)
    new_pr_branch_name: str = "PR_BRANCH"
    fetch_cmd: List[str] = [
        "git",
        "fetch",
        "origin",
        f"pull/{github_id}/head:{new_pr_branch_name}",
    ]
    _run_subprocess_log(fetch_cmd)
    os.chdir(old_cwd)
    git_checkout_branch(path_to_repo, new_pr_branch_name)


def run_workflow(
    lute_location: str,
    config_file: str,
    workflow_file: str,
    use_test_inst: bool = False,
    is_admin: bool = False,
) -> bool:
    """Run a workflow.

    Args:
        lute_location (str): Path to the LUTE installation.

        config_file (str): Path to the configuration YAML.

        workflow_file (str): Path to the DAG definition YAML.

        use_test_inst (bool): Whether to use the test Airflow instance (instead
            of production instance). Default: False.

        is_admin (bool): Whether running as administrator account. Default: False.

    Returns:
        is_successful (bool): True if workflow returns successful. False otherwise.
    """
    airflow_instance: str
    instance_str: str
    if use_test_inst:
        airflow_instance = "http://172.24.5.190:8080"
        instance_str = "test"
    else:
        airflow_instance = "http://172.24.5.247:8080"
        instance_str = "prod"

    wf_name: str = "test_dynamic"
    airflow_api_endpoints: Dict[str, str] = {
        "health": "api/v1/health",
        "run_dag": f"api/v1/dags/lute_{wf_name}/dagRuns",
        "get_tasks": f"api/v1/dags/lute_{wf_name}/tasks",
        "get_xcom": (  # Need to format dag_run_id, task_id, xcom_key
            f"api/v1/dags/lute_{wf_name}/dagRuns/{{dag_run_id}}/taskInstances"
            f"/{{task_id}}/xcomEntries/{{xcom_key}}"
        ),
        # Only for User-Specified workflows
        "mod_dag": f"api/v1/dags/lute_{wf_name}",  # Delete, pause/unpause, etc.
        "create_defn": "api/v1/variables",
        "update_defn": "api/v1/variables/user_workflow",
        "parse_file": "api/v1/parseDagFile/{file_token}",
    }

    pw: str = _retrieve_pw(instance_str, is_admin=is_admin)
    user_name: str = "btx" if is_admin else "lcls_user"
    auth: HTTPBasicAuth = HTTPBasicAuth(user_name, pw)
    resp: requests.models.Response = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['health']}",
        auth=auth,
    )
    resp.raise_for_status()

    params: Dict[str, Union[str, bool]] = {
        "config_file": config_file,
        "debug": True,
    }

    wf_defn: Dict[str, Any] = {}

    if not os.path.exists(workflow_file):
        logger.error("Workflow definition path does not exist! Exiting!")
        sys.exit(-1)
    with open(workflow_file, "r") as f:
        wf_defn = yaml.load(f, yaml.FullLoader)

    # Update user workflow definition in Airflow
    new_workflow: Dict[str, str] = {
        "key": "user_workflow",
        "value": json.dumps(wf_defn),
    }
    resp = requests.patch(
        f"{airflow_instance}/{airflow_api_endpoints['update_defn']}",
        json=new_workflow,
        auth=auth,
    )
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 404:
            # Workflow definition not found so previous DAG completed properly
            resp = requests.post(
                f"{airflow_instance}/{airflow_api_endpoints['create_defn']}",
                json=new_workflow,
                auth=auth,
            )
            resp.raise_for_status()
        else:
            raise
    # Let's wait some time and update again...
    # Try and allow Airflow to get used to the Dynamic DAG it doesn't like...
    time.sleep(2)
    resp = requests.patch(
        f"{airflow_instance}/{airflow_api_endpoints['update_defn']}",
        json=new_workflow,
        auth=auth,
    )
    resp.raise_for_status()
    logger.debug("Sent new workflow definition.")
    resp = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['mod_dag']}",
        auth=auth,
    )
    resp.raise_for_status()
    file_token: str = resp.json()["file_token"]
    f_endpoint: str = airflow_api_endpoints["parse_file"].format(file_token=file_token)
    resp = requests.put(f"{airflow_instance}/{f_endpoint}", auth=auth)
    resp.raise_for_status()
    logger.debug("Re-parsed DAG for setup with new workflow.")

    # Experiment, run #, and ARP env variables come from ARP submission only
    # We override above or exit if we cannot, so we cast here
    assert isinstance(os.getenv("EXPERIMENT"), str)
    assert isinstance(os.getenv("RUN_NUM"), str)
    assert isinstance(os.getenv("ARP_JOB_ID"), str)
    assert isinstance(os.getenv("Authorization"), str)
    dag_run_data: DagRunData = {
        "dag_run_id": str(uuid.uuid4()),
        "conf": {
            "experiment": cast(str, os.getenv("EXPERIMENT")),
            "run_id": f"{cast(str, os.getenv('RUN_NUM'))}_{datetime.datetime.utcnow().isoformat()}",
            "JID_UPDATE_COUNTERS": os.getenv("JID_UPDATE_COUNTERS"),
            "ARP_ROOT_JOB_ID": cast(str, os.getenv("ARP_JOB_ID")),
            "ARP_LOCATION": os.getenv("ARP_LOCATION", "S3DF"),
            "Authorization": cast(str, os.getenv("Authorization")),
            "user": getpass.getuser(),
            "lute_location": lute_location,
            "kerb_file": cache_file,
            "lute_params": params,
            "slurm_params": extra_args,
            "workflow": wf_defn,  # Only used for custom defined workflows.
        },
    }

    # Get Task information
    task_ids: List[str]
    # Airflow shouldn't have list of Tasks yet so we parse manually
    task_ids = ["retrieve_workflow"]

    def get_names(wf_defn: Dict[str, Any], names: List[str]) -> None:
        names.append(f"user_workflow.{wf_defn['task_name']}")
        for wf_new in wf_defn["next"]:
            get_names(wf_new, names)

    get_names(wf_defn, task_ids)
    task_ids = sorted(task_ids)
    task_id_str: str = ",\n\t- ".join(tid for tid in task_ids)
    logger.info(
        f"Contains Managed Tasks (alphabetical, not execution order):\n\t- {task_id_str}"
    )

    # Submit hopefully Airflow will do okay
    dag_run_data["dag_run_id"] = str(uuid.uuid4())
    resp = requests.post(
        f"{airflow_instance}/{airflow_api_endpoints['run_dag']}",
        json=dag_run_data,
        auth=auth,
    )
    resp.raise_for_status()
    dag_run_id: str = dag_run_data["dag_run_id"]
    logger.info(f"Submitted DAG (Workflow): {wf_name}\nDAG_RUN_ID: {dag_run_id}")
    dag_state: str = resp.json()["state"]
    logger.info(f"DAG is {dag_state}")
    # Enter loop for checking status
    time.sleep(1)
    # Same as run_dag endpoint, but needs to include the dag_run_id on the end
    url: str = f"{airflow_instance}/{airflow_api_endpoints['run_dag']}/{dag_run_id}"
    # Pulling logs for each Task via XCom
    xcom_key: str = "log"
    completed_tasks: Dict[str, str] = {}  # Remember exit status of each Task
    logged_running: List[str] = []  # Keep track to only print "running" once
    while True:
        time.sleep(1)
        # DAG Status
        resp = requests.get(url, auth=auth)
        resp.raise_for_status()
        dag_state = resp.json()["state"]
        # Check Task instances
        task_url: str = f"{url}/taskInstances"
        resp = requests.get(task_url, auth=auth)
        resp.raise_for_status()
        instance_information: List[Dict[str, Any]] = resp.json()["task_instances"]
        for inst in instance_information:
            task_id: str = inst["task_id"]
            task_state: Optional[str] = inst["state"]
            if task_id not in completed_tasks and task_state not in (
                None,
                "scheduled",
                "queued",
                "removed",
            ):
                if task_id not in logged_running:
                    # Should be "running" by first time it reaches here.
                    # Or e.g. "upstream_failed"... Setup to skip "scheduled"
                    logger.info(f"{task_id} state: {task_state}")
                    logged_running.append(task_id)

                if task_state in ("success", "failed"):
                    # Only pushed to XCOM at the end of each Task
                    xcom_url: str = (
                        f"{airflow_instance}/{airflow_api_endpoints['get_xcom']}"
                    )
                    xcom_url = xcom_url.format(
                        dag_run_id=dag_run_id,
                        task_id=task_id,
                        xcom_key=xcom_key,
                    )
                    try:
                        resp = requests.get(xcom_url, auth=auth)
                        resp.raise_for_status()
                        logs: str = resp.json()["value"]  # Only want to print once.
                        logger.info(f"Providing logs for {task_id}")
                        print("-" * 50, flush=True)
                        print(logs, flush=True)
                        print("-" * 50, flush=True)
                    except HTTPError:
                        # retrieve_workflow has no logs...
                        logger.info(f"No logs for {task_id}.")
                    logger.info(f"End of logs for {task_id}")
                    completed_tasks[task_id] = task_state

                # Ignore type check since cannot be None here
                elif task_state in ("upstream_failed"):  # type: ignore
                    # upstream_failed never launches so has no log
                    completed_tasks[task_id] = task_state  # type: ignore

        if dag_state in ("queued", "running"):
            continue
        logger.info(f"DAG exited: {dag_state}")
        break
    if dag_state == "failed":
        return False
    else:
        return True


def clean_up(
    cache_file: Optional[str], lute_location: str, output_location: str
) -> None:
    # We had to do some funny business to get Kerberos credentials...
    # Cleanup now that we're done
    logger.debug("Removing duplicate Kerberos credentials.")
    # This should be defined if we get here
    # Format is FILE:/.../...
    if cache_file is not None:
        try:
            os.remove(cache_file[5:])
            os.rmdir(f"{os.path.expanduser('~')}/.tmp_cache")
        except FileNotFoundError:
            logger.error("No cache file found to remove.")

    logger.info(f"Cleaning up {lute_location}")
    shutil.rmtree(lute_location)
    logger.info(f"Cleaning up {output_location}")
    shutil.rmtree(output_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="functional_airflow_test_suite",
        description="Run the LUTE functional test suite.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-a",
        "--admin",
        help="Run as Airflow admin. Requires permissions.",
        action="store_true",
    )
    # Options for running specific versions of LUTE
    parser.add_argument(
        "--git_pr_id",
        help="Check out a specific GitHub PR ID of LUTE to run (a PR branch).",
        type=int,
    )
    parser.add_argument(
        "--git_tag",
        help="Check out a specific git tag of LUTE to run (e.g. a release).",
        type=str,
    )
    parser.add_argument(
        "--no_delete",
        help="If passed, do not delete output files when tests are finished.",
        action="store_true",
    )
    parser.add_argument(
        "-r", "--run_dir", help="Directory to install LUTE to.", type=str, required=True
    )
    parser.add_argument(
        "--run_tests",
        help=(
            "Provide a comma-separated string of tests to run.  If provided, this "
            "script will only run those, rather than the default behaviour of "
            "running all tests. E.g: --run_these_tests test2,test5. Tests that do not "
            "exist are silently ignored."
        ),
    )
    parser.add_argument(
        "--tests_dir",
        help=(
            "Specify an alternative path to tests than those from the LUTE clone.\n"
            "Must have the same directory structure: $DIR/test1/... $DIR/test2/...\n"
            "If this flag and --use_local_tests are both passed, this one is used."
        ),
        type=str,
    )
    parser.add_argument(
        "--test_airflow", help="Use test Airflow instance.", action="store_true"
    )
    parser.add_argument(
        "--use_local_tests",
        help=(
            "Use the tests from the installation of LUTE where this script is called,\n"
            "rather than those from the clone of LUTE which is run against, or another\n"
            "directory if passed. If this flag and --tests_dir are both passed, "
            "--tests_dir is used."
        ),
        action="store_true",
    )

    args: argparse.Namespace
    extra_args: List[str]
    args, extra_args = parser.parse_known_args()

    cache_file: Optional[str] = os.getenv("KRB5CCNAME")
    if cache_file is None:
        logger.error("No Kerberos cache. Try running `kinit` and resubmitting.")
        sys.exit(-1)

    logger.debug(f"Cloning LUTE to {args.run_dir}")
    git_clone("slac-lcls/lute", args.run_dir, 0o777)
    if args.git_tag is not None and args.git_pr_id is not None:
        logger.warning(
            "Provided both a git tag and git ID to use. Will default to using the ID."
        )
        logger.info(f"Switching to PR branch ID {args.git_pr_id}")
        git_fetch_pr_branch(f"{args.run_dir}/lute", args.git_pr_id)
    elif args.git_tag is not None:
        logger.info(f"Switching to tag {args.git_tag}")
        git_checkout_branch(f"{args.run_dir}/lute", args.git_tag)
    elif args.git_pr_id is not None:
        logger.info(f"Switching to PR branch ID {args.git_pr_id}")
        git_fetch_pr_branch(f"{args.run_dir}/lute", args.git_pr_id)
    else:
        logger.info("Running LUTE from dev branch.")

    lute_location: str = f"{args.run_dir}/lute"

    output_location: str = f"{args.run_dir}/lute_output"
    logger.info(f"Will write output to {output_location}")
    os.makedirs(output_location, mode=0o777)
    os.chmod(output_location, mode=0o777)

    func_tests_dir: str
    if args.tests_dir is not None:
        if args.use_local_tests:
            logger.warning(
                "Provided both `--tests_dir` and `--use_local_tests`. Will use `--tests_dir`."
            )
        func_tests_dir = args.tests_dir
    elif args.use_local_tests:
        func_tests_dir = f"{os.path.dirname(__file__)}/functional"
    else:
        func_tests_dir = f"{lute_location}/tests/functional"

    def use_test(
        test_dir: str, func_tests_dir: str, usable_tests_str: Optional[str] = None
    ) -> bool:
        if usable_tests_str is not None:
            usable_dirs: List[str] = [
                f"{func_tests_dir}/{d}" for d in usable_tests_str.split(",")
            ]
            return test_dir in usable_dirs
        else:
            return test_dir != func_tests_dir

    test_dirs: List[str] = [
        x[0]
        for x in os.walk(func_tests_dir)
        if use_test(x[0], func_tests_dir, args.run_tests)
    ]
    logger.info(f"Will attempt running {len(test_dirs)} tests")

    num_successful: int = 0
    num_unsuccessful: int = 0
    try:
        for test_dir in test_dirs:
            test_name: str = test_dir.split("/")[-1]
            logger.info(f"Running test: {test_name}")
            # We assume that each test directory has a config.yaml and dag.yaml
            config_file: str = f"{test_dir}/config.yaml"
            sed_pattern: str = f's|work_dir:\(.*\)|work_dir: \\"{output_location}\\"|g'
            inplace_sed(config_file, sed_pattern)
            wf_file: str = f"{test_dir}/dag.yaml"
            should_fail: bool = (
                True if os.path.exists(f"{test_dir}/SHOULD_FAIL") else False
            )

            # Retrieve the experiment and run from each test YAML
            experiment: str = parse_yaml_value(config_file, "experiment")
            run: str = parse_yaml_value(config_file, "run")

            # Setup environment variables -> These are passed to airflow
            # run_workflow function uses them
            os.environ["EXPERIMENT"] = experiment
            os.environ["RUN_NUM"] = run
            os.environ["Authorization"] = _request_arp_token(experiment)
            os.environ["ARP_JOB_ID"] = str(uuid.uuid4())

            is_successful: bool = run_workflow(
                lute_location=lute_location,
                config_file=config_file,
                workflow_file=wf_file,
                use_test_inst=args.test_airflow,
                is_admin=args.admin,
            )

            if is_successful:
                if should_fail:
                    num_unsuccessful += 1
                    logger.info(
                        f"Test workflow {test_name} completed successfully but should fail!"
                    )
                else:
                    num_successful += 1
                    logger.info(f"Test workflow {test_name} completed successfully.")
            else:
                if should_fail:
                    num_successful += 1
                    logger.info(
                        f"Test workflow {test_name} was unsuccessful but this is marked as intentional."
                    )
                else:
                    num_unsuccessful += 1
                    logger.error(f"Test workflow {test_name} was unsuccessfull!")
    except Exception as e:
        logger.error(f"Error in testing framework: {e}")
        if not args.no_delete:
            clean_up(cache_file, lute_location, output_location)
        sys.exit(-1)

    logger.info(f"Ran {len(test_dirs)} tests. {num_successful} were successful.")
    if not args.no_delete:
        clean_up(cache_file, lute_location, output_location)
