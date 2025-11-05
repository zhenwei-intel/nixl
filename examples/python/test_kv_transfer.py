import time
import numpy as np
import torch
import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

def init_agent(name):
    config = nixl_agent_config(backends=["UCX"])
    return nixl_agent(name, config)

def create_descs(addr_base, num_descs, length):
    descs = np.zeros((num_descs, 3), dtype=np.uint64)
    indices = np.arange(num_descs)
    descs[:, 0] = addr_base + indices * length
    descs[:, 1] = length
    descs[:, 2] = 0
    return descs, indices

def register_memory(agent, descs, mem_type="DRAM"):
    reg_descs = agent.get_reg_descs(descs, mem_type)
    assert agent.register_memory(reg_descs) is not None
    return agent.get_xfer_descs(descs, mem_type)

def main():
    num_elements = 100
    num_descs = 1
    element_size = 4  # float32
    total_length = num_elements * element_size

    # Source tensor and agent
    src_tensor = torch.ones(num_elements, dtype=torch.float32)
    src_agent = init_agent("prefill")
    src_descs, src_indices = create_descs(src_tensor.data_ptr(), num_descs, total_length)
    src_xfer_descs = register_memory(src_agent, src_descs)

    # Destination tensor and agent
    dst_tensor = torch.zeros(num_elements, dtype=torch.float32)
    dst_agent = init_agent("decode")
    dst_descs, dst_indices = create_descs(dst_tensor.data_ptr(), num_descs, total_length)
    dst_xfer_descs = register_memory(dst_agent, dst_descs)

    # Setup remote agent and transfer handles
    src_metadata = src_agent.get_agent_metadata()
    remote_agent_name = dst_agent.add_remote_agent(src_metadata)
    local_prep_handle = dst_agent.prep_xfer_dlist("NIXL_INIT_AGENT", dst_xfer_descs)
    remote_prep_handle = dst_agent.prep_xfer_dlist(remote_agent_name, src_xfer_descs)

    xfer_handle = dst_agent.make_prepped_xfer(
        "READ",
        local_prep_handle,
        dst_indices,
        remote_prep_handle,
        src_indices,
        b"UUID2"
    )
    assert xfer_handle
    print(xfer_handle)
    dst_agent.transfer(xfer_handle)
    time.sleep(10)
    print(dst_tensor) 

if __name__ == "__main__":
    main()