from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


class LuteParams(TypedDict):
    config_file: str
    debug: bool


class FlowConf(TypedDict):
    experiment: str
    run_id: str
    JID_UPDATE_COUNTERS: Optional[str]
    ARP_ROOT_JOB_ID: str
    ARP_LOCATION: str
    Authorization: str
    user: str
    lute_location: str
    kerb_file: Optional[str]
    lute_params: LuteParams
    slurm_params: List[str]
    workflow: Dict[str, Any]
    run_type: Optional[str]
    is_daq2: Optional[bool]


class FlowRequestDict(TypedDict):
    parameters: Dict[Literal["flow_conf"], FlowConf]
