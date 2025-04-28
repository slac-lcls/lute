"""Functions containing more complex logic for Task environment configuration.

Functions:
    setup_smd2_env(): Sets up psana2 environment variables.
"""

__all__ = ["setup_smd2_env"]
__author__ = "Gabriel Dorlhiac"

import os
import subprocess
from typing import List, Dict, Tuple, Optional


def setup_smd2_env() -> Dict[str, str]:
    """Setup environment variables smalldata_tools uses with psana2."""
    # partition: str = ...
    psana_vars: Dict[str, str] = {}
    nodes: Optional[str] = os.getenv("SLURM_NNODES")
    cores_per_node: Optional[str] = os.getenv("SLURM_NTASKS_PER_NODE")
    # If above are None, not running in SLURM
    if nodes is None or cores_per_node is None:
        psana_vars["PS_SRV_NODES"] = "1"
        psana_vars["PS_EB_NODES"] = "1"
        return psana_vars
    mpi_slots: int = int(cores_per_node) * int(nodes) - 1
    default_srv_cores: int = 16 * int(nodes)

    srv_cores: int = default_srv_cores

    default_eb_cores: int = (mpi_slots - srv_cores) // 16
    eb_cores: int = default_eb_cores

    psana_vars["PS_SRV_NODES"] = str(srv_cores)
    psana_vars["PS_EB_NODES"] = str(eb_cores)

    slurm_job_nodelist: Optional[str] = os.getenv("SLURM_JOB_NODELIST")
    if slurm_job_nodelist is None:
        return psana_vars
    cmd: List[str] = ["scontrol", "show", "hostnames", slurm_job_nodelist]
    host_list_bytes: bytes
    host_list_bytes, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()

    host_list: List[str] = host_list_bytes.decode().split("\n")[:-1]

    slurm_job_id: Optional[str] = os.getenv("SLURM_JOB_ID")
    if slurm_job_id is None:
        return psana_vars
    host_file: str = f"slurm_host_{slurm_job_id}"
    with open(host_file, "w") as f:
        for i in range(len(host_list)):
            if i == 0:
                f.write(f"{host_list[i]} slots=1")
            else:
                f.write(f"{host_list[i]}")

    # cpus_on_node: Optional[str] = os.getenv("SLURM_CPUS_ON_NODE")
    # Same as cores_per_node above
    # slurm_job_num_nodes: Optional[str] = os.getenv("SLURM_JOB_NUM_NODES")
    # Same as nodes as above

    n_ranks: int = int(cores_per_node) * (int(nodes) - 1) + 1

    psana_vars["PS_HOST_FILE"] = host_file
    psana_vars["PS_N_RANKS"] = str(n_ranks)

    return psana_vars
