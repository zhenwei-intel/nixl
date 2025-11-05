import time

import numpy as np

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger
import time
logger = get_logger(__name__)


def init_agent(name):
    agent_config = nixl_agent_config(backends=["UCX"])
    agent = nixl_agent(name, agent_config)
    return agent


num_descs = 2**8
length = 1024
addr_base = nixl_utils.malloc_passthru(num_descs * length)
logger.info(
    "Performance test: Creating nixlXferDList with %d descriptors", num_descs
)
prefill_node = init_agent("prefill")
descs_np = np.zeros((num_descs, 3), dtype=np.uint64)
indices = np.arange(num_descs)
descs_np[:, 0] = addr_base + indices * length
descs_np[:, 1] = length
descs_np[:, 2] = 0

xfer_dlist_prefill = prefill_node.get_xfer_descs(descs_np, "DRAM")
reg_dlist = prefill_node.get_reg_descs(descs_np, "DRAM")
assert prefill_node.register_memory(reg_dlist) is not None

decode_node = init_agent("decode")
addr_base2 = nixl_utils.malloc_passthru(num_descs * length)
descs_np2 = np.zeros((num_descs, 3), dtype=np.uint64)
indices2 = np.arange(num_descs)
descs_np2[:, 0] = addr_base2 + indices2 * length
descs_np2[:, 1] = length
descs_np2[:, 2] = 0

xfer_dlist_decode = decode_node.get_xfer_descs(descs_np2, "DRAM")
reg_dlist2 = decode_node.get_reg_descs(descs_np2, "DRAM")
assert decode_node.register_memory(reg_dlist2) is not None

# get remote agent metadata and xfer_dlist_prefill
agent_metadata = prefill_node.get_agent_metadata()
remote_agent_name = decode_node.add_remote_agent(agent_metadata)
local_prep_handle = decode_node.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_dlist_decode)
remote_prep_handle = decode_node.prep_xfer_dlist(remote_agent_name, xfer_dlist_prefill)

xfer_handle = decode_node.make_prepped_xfer(
    "READ",
    local_prep_handle,
    indices,
    remote_prep_handle,
    indices2,
    b"UUID2"
)
print(xfer_handle)
decode_node.transfer(xfer_handle)
time.sleep(10)
print(descs_np2)
assert xfer_handle