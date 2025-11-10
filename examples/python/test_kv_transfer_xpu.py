"""
DRAM

FI_PROVIDER=verbs python test_kv_transfer.py
2025-11-06 22:28:01 NIXL INFO    _api.py:361 Backend LIBFABRIC was instantiated
2025-11-06 22:28:01 NIXL INFO    _api.py:251 Initialized NIXL agent: prefill
2025-11-06 22:28:03 NIXL INFO    _api.py:361 Backend LIBFABRIC was instantiated
2025-11-06 22:28:03 NIXL INFO    _api.py:251 Initialized NIXL agent: decode
nixl_xfer_handle(0x5c161b0ff6a0, released=False)
Transfer done
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
        28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
        42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
        56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
        70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 80., 81., 82., 83.,
        84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97.,
        98., 99.])


FI_PROVIDER=tcp python test_kv_transfer.py
2025-11-06 22:37:57 NIXL INFO    _api.py:361 Backend LIBFABRIC was instantiated
2025-11-06 22:37:57 NIXL INFO    _api.py:251 Initialized NIXL agent: prefill
2025-11-06 22:38:39 NIXL INFO    _api.py:361 Backend LIBFABRIC was instantiated
2025-11-06 22:38:39 NIXL INFO    _api.py:251 Initialized NIXL agent: decode
nixl_xfer_handle(0x61f9eb6e6510, released=False)
Transfer done
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
        28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
        42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
        56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
        70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 80., 81., 82., 83.,
        84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97.,
        98., 99.])

VRAM
FI_PROVIDER=verbs HMEM_IFACE=ze python test_kv_transfer.py
2025-11-06 23:01:39 NIXL INFO    _api.py:361 Backend LIBFABRIC was instantiated
2025-11-06 23:01:39 NIXL INFO    _api.py:251 Initialized NIXL agent: prefill
2025-11-06 23:01:48 NIXL INFO    _api.py:361 Backend LIBFABRIC was instantiated
2025-11-06 23:01:48 NIXL INFO    _api.py:251 Initialized NIXL agent: decode
W1106 23:01:48.712733    8597 libfabric_topology.cpp:164] No NICs found for GPU 1, returning all devices
nixl_xfer_handle(0x5e12248cdf40, released=False)
Transfer done
tensor([ 0.0000e+00,  1.0000e+00,  2.0000e+00,  3.0000e+00,  4.0000e+00,
         5.0000e+00,  6.0000e+00,  7.0000e+00,  8.0000e+00,  9.0000e+00,
         1.0000e+01,  1.1000e+01,  1.2000e+01,  1.3000e+01,  1.4000e+01,
         1.5000e+01,  1.6000e+01,  1.7000e+01,  1.8000e+01,  1.9000e+01,
         2.0000e+01,  2.1000e+01,  2.2000e+01,  2.3000e+01,  2.4000e+01,
         2.5000e+01,  2.6000e+01,  2.7000e+01,  2.8000e+01,  2.9000e+01,
         3.0000e+01,  3.1000e+01,  3.2000e+01,  3.3000e+01,  3.4000e+01,
         3.5000e+01,  3.6000e+01,  3.7000e+01,  3.8000e+01,  3.9000e+01,
         4.0000e+01,  4.1000e+01,  4.2000e+01,  4.3000e+01,  4.4000e+01,
         4.5000e+01,  4.6000e+01,  4.7000e+01,  4.8000e+01,  4.9000e+01,
         5.0000e+01,  5.1000e+01,  5.2000e+01,  5.3000e+01,  5.4000e+01,
         5.5000e+01,  5.6000e+01,  5.7000e+01,  5.8000e+01,  5.9000e+01,
         6.0000e+01,  6.1000e+01,  6.2000e+01,  6.3000e+01,  7.0508e-30,
        -1.9945e+36,  1.9661e+05,  2.0158e+31, -5.0352e+17,  4.6994e+23,
         3.5624e-40,  1.0385e+34,  2.4887e-38,  1.9269e-34,  1.9819e-16,
        -1.6529e+29,  2.8469e-39, -2.1287e+00,  5.6575e-34,  3.4742e+28,
         6.1631e-24,  1.3166e-36,  2.4887e-38,  1.9269e-34,  1.5124e-39,
         2.1040e+23,  2.8470e-39, -2.0001e+00,  5.6575e-34,  7.8439e-03,
         1.1210e-44, -4.0253e+31,  1.0019e+34,  3.1421e+35, -1.7298e+09,
         3.3470e-20,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
       device='xpu:1')
free(): double free detected in tcache 2
Aborted (core dumped)


"""
import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

def init_agent(name):
    config = nixl_agent_config(backends=["LIBFABRIC"])
    return nixl_agent(name, config)

def create_descs(addr_base, num_descs, length, device_id=0):
    descs = np.zeros((num_descs, 3), dtype=np.uint64)
    indices = np.arange(num_descs)

    # 确保所有运算都使用 uint64
    addr_base = np.uint64(addr_base)
    length = np.uint64(length)
    descs[:, 0] = addr_base + indices * length
    descs[:, 1] = length
    descs[:, 2] = device_id
    return descs, indices

def register_memory(agent, descs, mem_type="DRAM"):
    reg_descs = agent.get_reg_descs(descs, mem_type)
    agent.register_memory(reg_descs, backends=["LIBFABRIC"])
    xfer_descs = agent.get_xfer_descs(descs, mem_type)
    return reg_descs, xfer_descs

def main():
    num_elements = 1024
    num_descs = 1
    element_size = 4  # float32
    total_length = num_elements * element_size

    # Source tensor and agent
    src_tensor = torch.arange(num_elements, dtype=torch.float32).to("xpu:0")
    src_agent = init_agent("prefill")
    src_descs, src_indices = create_descs(src_tensor.data_ptr(), num_descs, total_length, device_id=0)
    src_reg_descs,src_xfer_descs = register_memory(src_agent, src_descs, mem_type="VRAM")

    # Destination tensor and agent
    dst_tensor = torch.zeros(num_elements, dtype=torch.float32).to("xpu:1")
    dst_agent = init_agent("decode")
    dst_descs, dst_indices = create_descs(dst_tensor.data_ptr(), num_descs, total_length, device_id=1)
    dst_reg_descs, dst_xfer_descs = register_memory(dst_agent, dst_descs, mem_type="VRAM")

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

    transfer_done = False
    while not transfer_done:
        state = dst_agent.check_xfer_state(xfer_handle)
        if state == "ERR":
            print("Transfer got to Error state.")
            exit()
        elif state == "DONE":
            transfer_done = True
            print("Transfer done")
    _ = dst_tensor.sum()
    print(dst_tensor)

    dst_agent.release_xfer_handle(xfer_handle)
    dst_agent.release_dlist_handle(local_prep_handle)
    dst_agent.release_dlist_handle(remote_prep_handle)
    dst_agent.remove_remote_agent(remote_agent_name)
    src_agent.deregister_memory(src_reg_descs)
    dst_agent.deregister_memory(dst_reg_descs)

if __name__ == "__main__":
    main()
