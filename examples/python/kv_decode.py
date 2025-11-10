"""
python examples/python/kv_decode.py --ip 0.0.0.0 --port 8400 --backend UCX --device cuda:1
"""
import socket
import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config
import zmq
import pickle
import argparse
import time

def init_agent(name, backend):
    config = nixl_agent_config(backends=[backend])
    return nixl_agent(name, config)

def create_descs(addr_base, num_descs, length, device_id=0):
    descs = np.zeros((num_descs, 3), dtype=np.uint64)
    indices = np.arange(num_descs)
    descs[:, 0] = addr_base + indices * length
    descs[:, 1] = length
    descs[:, 2] = device_id
    return descs, indices

def register_memory(agent, descs, mem_type="DRAM", backend="UCX"):
    reg_descs = agent.get_reg_descs(descs, mem_type)
    assert agent.register_memory(reg_descs, backends=[backend]) is not None
    return agent.get_xfer_descs(descs, mem_type)

def main():
    parser = argparse.ArgumentParser(description="KV decode")
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Server IP address')
    parser.add_argument('--port', type=int, default=50007, help='Server port')
    parser.add_argument('--backend', type=str, default='UCX', help='Backend to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()

    num_elements = 100
    num_descs = 1
    element_size = 4  # float32
    total_length = num_elements * element_size
    if args.device == 'cpu':
        mem_type = "DRAM"
        device_id = 0
    else:
        mem_type = "VRAM"
        device_id = int(args.device.split(":")[1])

    # Server tensor and agent
    tensor = torch.zeros(num_elements, dtype=torch.float32).to(args.device)
    agent = init_agent("decode", backend=args.backend)
    descs, indices = create_descs(tensor.data_ptr(), num_descs, total_length, device_id=device_id)
    xfer_descs = register_memory(agent, descs, mem_type=mem_type, backend=args.backend)

    context = zmq.Context()
    socket_zmq = context.socket(zmq.REQ)
    zmq_addr = f"tcp://{args.ip}:{args.port}"
    connected = False
    while not connected:
        try:
            socket_zmq.connect(zmq_addr)
            print(f"[Decode] Connected to server at {zmq_addr}")
            connected = True
        except Exception as e:
            print(f"[Decode] Connection failed: {e}, retrying...")
            time.sleep(1)
    # get metadata
    socket_zmq.send(b"get_metadata")
    picked_info = socket_zmq.recv()
    prefill_info = pickle.loads(picked_info)
    print("[Decode] Received prefill_info:", prefill_info)
    remote_agent_name = agent.add_remote_agent(prefill_info["metadata"])
    local_prep_handle = agent.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_descs)
    remote_prep_handle = agent.prep_xfer_dlist(remote_agent_name, agent.get_xfer_descs(prefill_info["xfer_descs"], mem_type))
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
    # Check transfer status
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
