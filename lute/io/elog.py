"""Provides utilities for communicating with the LCLS eLog.

Make use of various eLog API endpoint to retrieve information or post results.

Functions:
    get_elog_auth(exp: str): Return an authorization object to interact with
        eLog API as an opr account for the hutch where `exp` was conducted.
    get_elog_http_request(url: str, request_type: str, **params): Make an HTTP
        request to the API endpoint at `url`.
    format_file_for_post(in_file: Union[str, tuple, list]): Prepare files
        according to the specification needed to add them as attachments to eLog
        posts.
    post_elog_message(exp: str, msg: str, tag: Optional[str],
                      title: Optional[str],
                      in_files: List[Union[str, tuple, list]],
                      auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Post a message to the eLog.
    post_elog_runtable(exp: str, run: int, data: Dict[str, Any],
                       auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Update run table in the eLog.
    get_elog_runs_by_tag(exp: str, tag: str,
                         auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Return a list of runs with a specific tag.
"""

__all__ = [
    "post_elog_message",
    "post_elog_runtable",
    "get_elog_runs_by_tag",
    "post_run_status",
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

BASE_URL: str = "https://pswww.slac.stanford.edu/ws-auth/lgbk/lgbk"


def get_elog_auth(exp: str) -> Union[HTTPBasicAuth, Dict]:
    """Produce authentication for the "opr" user associated to an experiment.

    Will return different authentication methods depending on whether the
    requested experiment is active or not.

    Args:
        exp (str): Name of the experiment to produce authentication for.

    Returns:
        auth (HTTPBasicAuth | Dict): HTTPBasicAuth for an active experiment
            based on username and password, or Kerberos headers for an inactive
            experiment.
    """
    # NEED TO FILL IN
    return HTTPBasicAuth("fake", "fake") or {}


def elog_http_request(
    url: str, request_type: str, **params
) -> Tuple[int, str, Optional[Any]]:
    """Make an HTTP request to the eLog.

    Args:
        url (str): eLog API endpoint.

        request_type (str): Type of request to make. Recognized options: POST or
            GET.
        **params (Dict): ALL parameters to pass with the HTTP request! Differs
            depending on the API endpoint.

    Returns:
        status_code (int): Response status code. Can be checked for errors.

        msg (str): An error message, or a message saying SUCCESS.

        value (Optional[Any]): For GET requests ONLY, return the requested
            information.
    """
    if request_type.upper() == "POST":
        resp: requests.models.Response = requests.post(url, **params)
    elif request_type.upper() == "GET":
        resp: requests.models.Response = requests.get(url, **params)
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


def post_run_status(
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

    auth: Union[HTTPBasicAuth, Dict] = auth or get_elog_auth(exp)
    post_url: str = f"{BASE_URL}/{exp}/ws/new_elog_entry"

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


def post_elog_runtable(
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
    auth: Union[HTTPBasicAuth, Dict] = auth or get_elog_auth(exp)

    params: Dict[str, Any] = {"params": {"run_num": run}, "json": data}

    if isinstance(auth, HTTPBasicAuth):
        params.update({"auth": auth})
    elif isinstance(auth, dict):
        params.update({"headers": auth})

    status_code, resp_msg, _ = elog_http_request(url=table_url, request_type="POST")

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
