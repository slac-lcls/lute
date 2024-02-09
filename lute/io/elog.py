"""Provides utilities for communicating with the LCLS eLog.

Make use of various eLog API endpoint to retrieve information or post results.

Functions:
    get_elog_opr_auth(exp: str): Return an authorization object to interact with
        eLog API as an opr account for the hutch where `exp` was conducted.

    get_elog_kerberos_auth(): Return the authorization headers for the user account
        submitting the job.

    elog_http_request(url: str, request_type: str, **params): Make an HTTP
        request to the API endpoint at `url`.

    format_file_for_post(in_file: Union[str, tuple, list]): Prepare files
        according to the specification needed to add them as attachments to eLog
        posts.

    post_elog_message(exp: str, msg: str, tag: Optional[str],
                      title: Optional[str],
                      in_files: List[Union[str, tuple, list]],
                      auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Post a message to the eLog.

    post_elog_run_status(data: Dict[str, Union[str, int, float]],
                         update_url: Optional[str] = None)
        Post a run status to the summary section on the Workflows>Control tab.

    post_elog_run_table(exp: str, run: int, data: Dict[str, Any],
                       auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Update run table in the eLog.

    get_elog_runs_by_tag(exp: str, tag: str,
                         auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Return a list of runs with a specific tag.

    get_elog_params_by_run(exp: str, params: List[str], runs: Optional[List[int]])
        Retrieve the requested parameters by run. If no run is provided, retrieve
        the requested parameters for all runs.
"""

__all__ = [
    "post_elog_message",
    "post_elog_run_table",
    "get_elog_runs_by_tag",
    "post_elog_run_status",
    "get_elog_params_by_run",
]
__author__ = "Gabriel Dorlhiac"

import sys
import requests
from requests.auth import HTTPBasicAuth
import socket
import mimetypes
import os
from typing import Any, Dict, Optional, List, Union, Tuple
from io import BufferedReader

from krtc import KerberosTicket


def get_elog_workflows(exp: str) -> Dict[str, str]:
    """Get the current workflow definitions for an experiment.

    Returns:
        defns (Dict[str, str]): A dictionary of workflow definitions.
    """
    raise NotImplementedError


def post_elog_workflow(
    exp: str,
    name: str,
    executable: str,
    wf_params: str,
    *,
    trigger: str = "run_end",
    location: str = "S3DF",
    **trig_args: str,
) -> None:
    """Create a new eLog workflow, or update an existing one.

    The workflow will run a specific executable as a batch job when the
    specified trigger occurs. The precise arguments may vary depending on the
    selected trigger type.

    Args:
        name (str): An identifying name for the workflow. E.g. "process data"

        executable (str): Full path to the executable to be run.

        wf_params (str): All command-line parameters for the executable as a string.

        trigger (str): When to trigger execution of the specified executable.
            One of:
                - 'manual': Must be manually triggered. No automatic processing.
                - 'run_start': Execute immediately if a new run begins.
                - 'run_end': As soon as a run ends.
                - 'param_is': As soon as a parameter has a specific value for a run.

        location (str): Where to submit the job. S3DF or NERSC.

        **trig_args (str): Arguments required for a specific trigger type.
            trigger='param_is' - 2 Arguments
                trig_param (str): Name of the parameter to watch for.
                trig_param_val (str): Value the parameter should have to trigger.
    """
    endpoint: str = f"{exp}/ws/create_update_workflow_def"
    trig_map: Dict[str, str] = {
        "manual": "MANUAL",
        "run_start": "START_OF_RUN",
        "run_end": "END_OF_RUN",
        "param_is": "RUN_PARAM_IS_VALUE",
    }
    if trigger not in trig_map.keys():
        raise NotImplementedError(
            f"Cannot create workflow with trigger type: {trigger}"
        )
    wf_defn: Dict[str, str] = {
        "name": name,
        "executable": executable,
        "parameters": wf_params,
        "trigger": trig_map[trigger],
        "location": location,
    }
    if trigger == "param_is":
        if "trig_param" not in trig_args or "trig_param_val" not in trig_args:
            raise RuntimeError(
                "Trigger type 'param_is' requires: 'trig_param' and 'trig_param_val' arguments"
            )
        wf_defn.update(
            {
                "run_param_name": trig_args["trig_param"],
                "run_param_val": trig_args["trig_param_val"],
            }
        )
    post_params: Dict[str, Dict[str, str]] = {"json": wf_defn}
    status_code, resp_msg, _ = elog_http_request(
        exp, endpoint=endpoint, request_type="POST", **post_params
    )


def get_elog_active_expmt(hutch: str, *, endstation: int = 0) -> str:
    """Get the current active experiment for a hutch.

    This function is the only function to manage the HTTP request independently.
    This is because it does not require an authorization object, and its result
    is needed for the generic function `elog_http_request` to work properly.

    Args:
        hutch (str): The hutch to get the active experiment for.

        endstation (int): The hutch endstation to get the experiment for. This
            should generally be 0.
    """

    base_url: str = "https://pswww.slac.stanford.edu/ws/lgbk/lgbk"
    endpoint: str = "ws/activeexperiment_for_instrument_station"
    url: str = f"{base_url}/{endpoint}"
    params: Dict[str, str] = {"instrument_name": hutch, "station": f"{endstation}"}
    resp: requests.models.Response = requests.get(url, params)
    if resp.status_code > 300:
        raise RuntimeError(
            f"Error getting current experiment!\n\t\tIncorrect hutch: '{hutch}'?"
        )
    if resp.json()["success"]:
        return resp.json()["value"]["name"]
    else:
        msg: str = resp.json()["error_msg"]
        raise RuntimeError(f"Error getting current experiment! Err: {msg}")


def get_elog_auth(exp: str) -> Union[HTTPBasicAuth, Dict[str, str]]:
    """Determine the appropriate auth method depending on experiment state.

    Returns:
        auth (HTTPBasicAuth | Dict[str, str]): Depending on whether an experiment
            is active/live, returns authorization for the hutch operator account
            or the current user submitting a job.
    """
    hutch: str = exp[:3]
    if exp.lower() == get_elog_active_expmt(hutch=hutch).lower():
        return get_elog_opr_auth(exp)
    else:
        return get_elog_kerberos_auth()


def get_elog_opr_auth(exp: str) -> HTTPBasicAuth:
    """Produce authentication for the "opr" user associated to an experiment.

    This method uses basic authentication using username and password.

    Args:
        exp (str): Name of the experiment to produce authentication for.

    Returns:
        auth (HTTPBasicAuth): HTTPBasicAuth for an active experiment based on
            username and password for the associated operator account.
    """
    # NEED TO FILL IN
    return HTTPBasicAuth("fake", "fake") or {}


def get_elog_kerberos_auth() -> Dict[str, str]:
    """Returns Kerberos authorization key.

    This functions returns authorization for the USER account submitting jobs.
    It assumes that `kinit` has been run.

    Returns:
        auth (Dict[str, str]): Dictionary containing Kerberos authorization key.
    """
    return KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()


def elog_http_request(
    exp: str, endpoint: str, request_type: str, **params
) -> Tuple[int, str, Optional[Any]]:
    """Make an HTTP request to the eLog.

    This method will determine the proper authorization method and update the
    passed parameters appropriately. Functions implementing specific endpoint
    functionality and calling this function should only pass the necessary
    endpoint-specific parameters and not include the authorization objects.

    Args:
        exp (str): Experiment.

        endpoint (str): eLog API endpoint.

        request_type (str): Type of request to make. Recognized options: POST or
            GET.

        **params (Dict): Endpoint parameters to pass with the HTTP request!
            Differs depending on the API endpoint. Do not include auth objects.

    Returns:
        status_code (int): Response status code. Can be checked for errors.

        msg (str): An error message, or a message saying SUCCESS.

        value (Optional[Any]): For GET requests ONLY, return the requested
            information.
    """
    auth: Union[HTTPBasicAuth, Dict[str, str]] = get_elog_auth(exp)
    base_url: str
    if isinstance(auth, HTTPBasicAuth):
        params.update({"auth": auth})
        base_url = "https://pswww.slac.stanford.edu/ws-auth/lgbk/lgbk"
    elif isinstance(auth, dict):
        params.update({"headers": auth})
        base_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk"

    url: str = f"{base_url}/{endpoint}"

    resp: requests.models.Response
    if request_type.upper() == "POST":
        resp = requests.post(url, **params)
    elif request_type.upper() == "GET":
        resp = requests.get(url, **params)
    else:
        return (-1, "Invalid request type!", None)

    status_code: int = resp.status_code
    msg: str = "SUCCESS"

    if resp.json()["success"] and request_type.upper() == "GET":
        return (status_code, msg, resp.json()["value"])

    if status_code >= 300:
        msg = f"Error when posting to eLog: Response {status_code}"

    if not resp.json()["success"]:
        err_msg = resp.json()["error_msg"]
        msg += f"\nInclude message: {err_msg}"
    return (resp.status_code, msg, None)


def format_file_for_post(
    in_file: Union[str, tuple, list]
) -> Tuple[str, Tuple[str, BufferedReader], Any]:
    """Format a file for attachment to an eLog post.

    The eLog API expects a specifically formatted tuple when adding file
    attachments. This function prepares the tuple to specification given a
    number of different input types.

    Args:
        in_file (str | tuple | list): File to include as an attachment in an
            eLog post.
    """
    # NEED TO FILL IN FOR VARIOUS INPUT TYPES
    out_file: Tuple[str, Tuple[str, BufferedReader], Any] = (
        "files",
        ("description", open("file", "rb")),
        mimetypes.guess_type("path_to_file")[0],
    )
    return out_file


def post_elog_run_status(
    data: Dict[str, Union[str, int, float]], update_url: Optional[str] = None
) -> None:
    """Post a summary to the status/report section of a specific run.

    In contrast to most eLog update/post mechanisms, this function searches
    for a specific environment variable which contains a temporary URL for
    posting. This is updated every job/run as jobs are submitted by the JID.
    The URL can optionally be passed to this function if it is known.

    Args:
        data (Dict[str, Union[str, int, float]]): The data to post to the eLog
            report section. Formatted in key:value pairs.

        update_url (Optional[str]): Optional update URL. If not provided, the
            function searches for the corresponding environment variable. If
            neither is found, the function aborts
    """
    post_list: List[Dict[str, str]] = [
        {"key": f"{key}", "value": f"{value}"} for key, value in data.items()
    ]

    if not update_url:
        update_url = os.environ.get("JID_UPDATE_COUNTERS")

    params: Dict[str, List[Dict[str, str]]] = {"json": post_list}
    if update_url:
        status_code, resp_msg, _ = elog_http_request(
            url=update_url, request_type="POST", **params
        )


def post_elog_message(
    exp: str,
    msg: str,
    *,
    tag: Optional[str],
    title: Optional[str],
    in_files: List[Union[str, tuple, list]],
    auth: Optional[Union[HTTPBasicAuth, Dict]] = None,
) -> Optional[str]:
    """Post a new message to the eLog. Inspired by the `elog` package.

    Args:
        exp (str): Experiment name.

        msg (str): BODY of the eLog post.

        tag (str | None): Optional "tag" to associate with the eLog post.

        title (str | None): Optional title to include in the eLog post.

        in_files (List[str | tuple | list]): Files to include as attachments in
            the eLog post.

        auth (None | HTTPBasicAuth | Dict): Authorization for the eLog API. Can
            be username/password or kerberos headers. If none are provided,
            authorization will be generated based on the experiment name.

    Returns:
        err_msg (str | None): If successful, nothing is returned, otherwise,
            return an error message.
    """
    # MOSTLY CORRECT
    out_files: list = []
    for f in in_files:
        out_files.append(format_file_for_post(in_file=f))
    post: Dict[str, str] = {}
    post["log_text"] = msg
    if tag:
        post["log_tags"] = tag
    if title:
        post["log_title"] = title

    auth: Union[HTTPBasicAuth, Dict] = auth or get_elog_opr_auth(exp)
    post_url: str = f"{BASE_URL}/{exp}/ws/new_elog_entry"
    # post_endpoint: str = f"{exp}/ws/new_elog_entry"

    params: Dict[str, Any] = {"data": post}

    if isinstance(auth, HTTPBasicAuth):
        params.update({"auth": auth})
    elif isinstance(auth, dict):
        params.update({"headers": auth})

    if out_files:
        params.update({"files": out_files})

    status_code, resp_msg, _ = elog_http_request(
        url=post_url, request_type="POST", **params
    )

    if resp_msg != "SUCCESS":
        return resp_msg
    # NEED to handle/propagate errors...


def post_elog_run_table(
    exp: str,
    run: int,
    data: Dict[str, Any],
    auth: Optional[Union[HTTPBasicAuth, Dict]] = None,
) -> Optional[str]:
    """Post data for eLog run tables.

    Args:
        exp (str): Experiment name.

        run (int): Run number corresponding to the data being posted.

        data (Dict[str, Any]): Data to be posted in format
            data["column_header"] = value.

        auth (None | HTTPBasicAuth | Dict): Authorization for the eLog API. Can
            be username/password or kerberos headers. If none are provided,
            authorization will be generated based on the experiment name.

    Returns:
        err_msg (None | str): If successful, nothing is returned, otherwise,
            return an error message.
    """
    table_url: str = f"{BASE_URL}/run_control/{exp}/ws/add_run_params"
    # endpoint: str = f"run_control/{exp}/ws/add_run_params"
    auth: Union[HTTPBasicAuth, Dict] = auth or get_elog_auth(exp)

    params: Dict[str, Any] = {"params": {"run_num": run}, "json": data}

    if isinstance(auth, HTTPBasicAuth):
        params.update({"auth": auth})
    elif isinstance(auth, dict):
        params.update({"headers": auth})

    status_code, resp_msg, _ = elog_http_request(
        exp, url=table_url, request_type="POST", **params
    )

    if resp_msg != "SUCCESS":
        return resp_msg
    # NEED to handle/propagate errors....


def get_elog_runs_by_tag(
    exp: str, tag: str, auth: Optional[Union[HTTPBasicAuth, Dict]] = None
) -> List[int]:
    """Retrieve run numbers with a specified tag.

    Args:
        exp (str): Experiment name.

        tag (str): The tag to retrieve runs for.

        auth (None | HTTPBasicAuth | Dict): Authorization for the eLog API. Can
            be username/password or kerberos headers. If none are provided,
            authorization will be generated based on the experiment name.
    """
    tag_url: str = f"{BASE_URL}/{exp}/ws/get_runs_with_tag?tag={tag}"
    params: Dict[str, Any] = {}
    if isinstance(auth, HTTPBasicAuth):
        params.update({"auth": auth})
    elif isinstance(auth, dict):
        params.update({"headers": auth})

    status_code, resp_msg, tagged_runs = elog_http_request(
        url=tag_url, request_type="GET", **params
    )

    if not tagged_runs:
        tagged_runs = []

    return tagged_runs


def get_elog_params_by_run(
    exp: str, params: List[str], runs: Optional[List[int]] = None
) -> Dict[str, str]:
    """Retrieve requested parameters by run or for all runs.

    Args:
        exp (str): Experiment to retrieve parameters for.

        params (List[str]): A list of parameters to retrieve. These can be any
            parameter recorded in the eLog (PVs, parameters posted by other
            Tasks, etc.)
    """
    ...
