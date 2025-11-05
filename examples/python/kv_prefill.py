import socket
import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config
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
    assert agent.register_memory(reg_descs) is not None
    return agent.get_xfer_descs(descs, mem_type)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KV Client")
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Server IP address')
    parser.add_argument('--port', type=int, default=50007, help='Server port')
    args = parser.parse_args()

    num_elements = 100
    num_descs = 1
    element_size = 4  # float32
    total_length = num_elements * element_size

    # Client tensor and agent
    tensor = torch.ones(num_elements, dtype=torch.float32, device="cuda:0")
    agent = init_agent("prefill")
    descs, indices = create_descs(tensor.data_ptr(), num_descs, total_length, device_id=0)
    xfer_descs = register_memory(agent, descs, mem_type="VRAM")

    # 连接 server，获取 metadata
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.ip, args.port))
        print(f"[Client] Connected to server at {args.ip}:{args.port}")
        metadata = s.recv(4096)
        # 发送自己的 agent metadata 和 xfer_descs
        my_info = {
            "metadata": agent.get_agent_metadata(),
            "xfer_descs": xfer_descs,
            "indices": indices
        }
        s.sendall(pickle.dumps(my_info))
        print("[Client] Sent metadata and xfer_descs to server")
        # 等待 server 完成 transfer
        # 可以根据需要实现进一步的同步或检查
    print("[Client] Tensor after transfer:", tensor)

if __name__ == "__main__":
    main()
