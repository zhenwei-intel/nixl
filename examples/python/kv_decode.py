import socket
import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config
import zmq
import pickle

def init_agent(name):
    config = nixl_agent_config(backends=["UCX"])
    return nixl_agent(name, config)

def create_descs(addr_base, num_descs, length, device_id=0):
    descs = np.zeros((num_descs, 3), dtype=np.uint64)
    indices = np.arange(num_descs)
    descs[:, 0] = addr_base + indices * length
    descs[:, 1] = length
    descs[:, 2] = device_id
    return descs, indices

def register_memory(agent, descs, mem_type="DRAM"):
    reg_descs = agent.get_reg_descs(descs, mem_type)
    assert agent.register_memory(reg_descs, backends=["UCX"]) is not None
    return agent.get_xfer_descs(descs, mem_type)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KV decode")
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Server IP address')
    parser.add_argument('--port', type=int, default=50007, help='Server port')
    args = parser.parse_args()

    num_elements = 100
    num_descs = 1
    element_size = 4  # float32
    total_length = num_elements * element_size

    # Server tensor and agent
    tensor = torch.zeros(num_elements, dtype=torch.float32, device="cuda:1")
    agent = init_agent("decode")
    descs, indices = create_descs(tensor.data_ptr(), num_descs, total_length, device_id=1)
    xfer_descs = register_memory(agent, descs, mem_type="VRAM")

    context = zmq.Context()
    socket_zmq = context.socket(zmq.REQ)
    zmq_addr = f"tcp://{args.ip}:{args.port}"
    socket_zmq.connect(zmq_addr)
    print(f"[Decode] Connected to server at {zmq_addr}")
    # 请求 metadata
    socket_zmq.send(b"get_metadata")
    picked_info = socket_zmq.recv()
    prefill_info = pickle.loads(picked_info)
    print("[Decode] Received prefill_info:", prefill_info)
    remote_agent_name = agent.add_remote_agent(prefill_info["metadata"])
    local_prep_handle = agent.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_descs)
    remote_prep_handle = agent.prep_xfer_dlist(remote_agent_name, agent.get_xfer_descs(prefill_info["xfer_descs"], "VRAM"))
    xfer_handle = agent.make_prepped_xfer(
        "READ",
        local_prep_handle,
        indices,
        remote_prep_handle,
        prefill_info["indices"],
        b"UUID2"
    )
    assert xfer_handle
    print("[Decode] xfer_handle:", xfer_handle)
    agent.transfer(xfer_handle)
    # 检查 transfer 状态
    transfer_done = False
    while not transfer_done:
        state = agent.check_xfer_state(xfer_handle)
        if state == "ERR":
            print("[Decode] Transfer got to Error state.")
            break
        elif state == "DONE":
            transfer_done = True
            print("[Decode] Transfer done")
    print("[Decode] Tensor after transfer:", tensor)

if __name__ == "__main__":
    main()
