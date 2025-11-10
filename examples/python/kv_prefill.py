"""
python examples/python/kv_prefill.py --ip 0.0.0.0 --port 8400 --timeout 3 --backend UCX --device cuda:0
"""
import socket
import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config
import pickle
import time
import zmq

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
    import argparse
    parser = argparse.ArgumentParser(description="KV Client")
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Server IP address')
    parser.add_argument('--port', type=int, default=50007, help='Server port')
    parser.add_argument('--backend', type=str, default='UCX', help='Backend to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--timeout', type=int, default=3600, help='Sleep time after prefill')
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

    # Client tensor and agent
    tensor = torch.arange(num_elements, dtype=torch.float32).to(args.device)
    agent = init_agent("prefill", backend=args.backend)
    descs, indices = create_descs(tensor.data_ptr(), num_descs, total_length, device_id=device_id)
    xfer_descs = register_memory(agent, descs, mem_type=mem_type, backend=args.backend)  # can not be omitted
    my_info = {
        "metadata": agent.get_agent_metadata(),
        "xfer_descs": descs,
        "indices": indices
    }
    print(my_info)
    packed_info = pickle.dumps(my_info)

    # start socket server，waiting client connection
    context = zmq.Context()
    socket_zmq = context.socket(zmq.REP)
    socket_zmq.bind(f"tcp://{args.ip}:{args.port}")
    print(f"[Server] Waiting for client at {args.ip}:{args.port} ...")
    msg = socket_zmq.recv()
    if msg == b"get_metadata":
        socket_zmq.send(packed_info)
    else:
        print(f"[Server] Unexpected message: {msg}")
        return

    time.sleep(args.timeout)  # can not be omitted
    print("[Server] Tensor after transfer:", tensor)

if __name__ == "__main__":
    main()
