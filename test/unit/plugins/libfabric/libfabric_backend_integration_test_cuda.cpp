/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Integration test for libfabric backend descriptor offset handling
 * Tests the actual backend with multiple descriptors pointing to different offsets
 * within the same registered memory region.


FI_PROVIDER=verbs ./test/unit/plugins/libfabric/test_libfabric_backend_integration --pthread

 */

#include <iostream>
#include <cassert>
#include <cstring>
#include <memory>
#include <unistd.h>

#include "libfabric_backend.h"
#include "common/nixl_log.h"

#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

int gpu_id = 0;

static void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

static int cudaQueryAddr(void *address, bool &is_dev,
                         CUdevice &dev, CUcontext &ctx)
{
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;

    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);

    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    return (CUDA_SUCCESS != result);
}

nixlLibfabricEngine *
createEngine(std::string name, bool p_thread) {
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;

    init.enableProgTh = p_thread;
    init.pthrDelay = 100;
    init.localAgent = name;
    init.customParams = &custom_params;
    init.type = "LIBFABRIC";

    auto engine = new nixlLibfabricEngine(&init);
    assert(!engine->getInitErr());
    if (engine->getInitErr()) {
        std::cout << "Failed to initialize libfabric engine" << std::endl;
        exit(1);
    }

    return engine;
}

void
releaseEngine(nixlLibfabricEngine *engine) {
    delete engine;
}


void
allocateAndRegister(nixlLibfabricEngine *engine,
                    int dev_id,
                    nixl_mem_t mem_type,
                    void *&addr,
                    size_t len,
                    nixlBackendMD *&md) {
    nixlBlobDesc desc;

    bool is_dev;
    CUdevice dev;
    CUcontext ctx;

    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
    checkCudaError(cudaMalloc(&addr, len), "Failed to allocate CUDA buffer 0");
    cudaQueryAddr(addr, is_dev, dev, ctx);
    std::cout << "CUDA addr: " << std::hex << addr << " dev=" << std::dec << dev
        << " ctx=" << std::hex << ctx << std::dec << std::endl;

    desc.addr = (uintptr_t)addr;
    desc.len = len;
    desc.devId = dev_id;

    int ret = engine->registerMem(desc, mem_type, md);
    assert(ret == NIXL_SUCCESS);
}

void
deallocateAndDeregister(nixlLibfabricEngine *engine,
                        int dev_id,
                        nixl_mem_t mem_type,
                        void *&addr,
                        nixlBackendMD *&md) {
    engine->deregisterMem(md);

    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
    checkCudaError(cudaFree(addr), "Failed to allocate CUDA buffer 0");
}

void
loadRemote(nixlLibfabricEngine *engine,
           int dev_id,
           std::string agent,
           nixl_mem_t mem_type,
           void *addr,
           size_t len,
           nixlBackendMD *&lmd,
           nixlBackendMD *&rmd) {
    nixlBlobDesc info;
    info.addr = (uintptr_t)addr;
    info.len = len;
    info.devId = dev_id;
    engine->getPublicData(lmd, info.metaInfo);

    assert(info.metaInfo.size() > 0);

    int ret = engine->loadRemoteMD(info, mem_type, agent, rmd);
    assert(NIXL_SUCCESS == ret);
}

void
populateDescs(nixl_meta_dlist_t &descs, int dev_id, void *addr, int desc_cnt, size_t desc_size,
              nixlBackendMD *&md) {
    for (int i = 0; i < desc_cnt; i++) {
        nixlMetaDesc req;
        req.addr = (uintptr_t)(((char *)addr) + i * desc_size); // Different offset per descriptor
        req.len = desc_size;
        req.devId = dev_id;
        req.metadataP = md;
        descs.addDesc(req);
    }
}

void
performTransfer(nixlLibfabricEngine *engine1,
                nixlLibfabricEngine *engine2,
                nixl_meta_dlist_t &req_src_descs,
                nixl_meta_dlist_t &req_dst_descs,
                void *addr1,
                void *addr2,
                size_t total_len,
                nixl_xfer_op_t op) {

    std::string remote_agent("Agent2");
    if (engine1 == engine2)
        remote_agent = "Agent1";

    std::cout << "\t" << (op == NIXL_READ ? "READ" : "WRITE") << " from " << addr1 << " to "
              << addr2 << " (" << total_len << " bytes, " << req_src_descs.descCount()
              << " descriptors)\n";

    nixl_opt_b_args_t opt_args;
    opt_args.hasNotif = false;

    // Prepare and post transfer
    nixlBackendReqH *handle = nullptr;
    nixl_status_t ret = engine1->prepXfer(op, req_src_descs, req_dst_descs, remote_agent, handle, &opt_args);
    assert(ret == NIXL_SUCCESS);

    ret = engine1->postXfer(op, req_src_descs, req_dst_descs, remote_agent, handle, &opt_args);
    assert(ret == NIXL_SUCCESS || ret == NIXL_IN_PROG);

    if (ret == NIXL_SUCCESS) {
        cout << "\t\tTransfer completed immediately\n";
    } else {
        cout << "\t\tWaiting for transfer completion...\n";
        while (ret == NIXL_IN_PROG) {
            ret = engine1->checkXfer(handle);
            // checkXfer() already progresses rails when progress thread is disabled
            assert(ret == NIXL_SUCCESS || ret == NIXL_IN_PROG);
        }
    }

    engine1->releaseReqH(handle);
    cout << "\t\tTransfer complete\n";
}

void
test_multi_descriptor_offsets(bool p_thread) {
    std::cout << "\n\n";
    std::cout << "****************************************************\n";
    std::cout << "   Multi-descriptor offset test (Integration)\n";
    std::cout << "   P-Thread=" << (p_thread ? "ON" : "OFF") << "\n";
    std::cout << "****************************************************\n";
    std::cout << "\n";

    std::string agent1("Agent1");
    std::string agent2("Agent2");

    // Create engines
    nixlLibfabricEngine *engine1 = createEngine(agent1, p_thread);
    nixlLibfabricEngine *engine2 = createEngine(agent2, p_thread);

    // Test parameters
    const size_t TOTAL_SIZE = 1024 * 1024; // 1MB total
    const size_t DESC_SIZE = 64 * 1024;    // 64KB per descriptor
    const int DESC_COUNT = TOTAL_SIZE / DESC_SIZE; // 16 descriptors

    std::cout << "Test configuration:\n";
    std::cout << "  Total buffer size: " << TOTAL_SIZE << " bytes\n";
    std::cout << "  Descriptor size: " << DESC_SIZE << " bytes\n";
    std::cout << "  Descriptor count: " << DESC_COUNT << "\n\n";

    // Allocate and register buffers
    void *send_buf = nullptr;
    void *recv_buf = nullptr;
    nixlBackendMD *send_md = nullptr;
    nixlBackendMD *recv_md = nullptr;

    allocateAndRegister(engine1, 0, VRAM_SEG, send_buf, TOTAL_SIZE, send_md);
    allocateAndRegister(engine2, 1, VRAM_SEG, recv_buf, TOTAL_SIZE, recv_md);

    // Fill send buffer with unique pattern for each descriptor's region
    for (int i = 0; i < DESC_COUNT; i++) {
        size_t offset = i * DESC_SIZE;
        uint8_t pattern = static_cast<uint8_t>(i);
        for (size_t j = 0; j < DESC_SIZE; j++) {
            ((uint8_t *)send_buf)[offset + j] = pattern;
        }
    }

    // Zero receive buffer
    memset(recv_buf, 0, TOTAL_SIZE);

    // Exchange connection info
    std::string conn1, conn2;
    engine1->getConnInfo(conn1);
    engine2->getConnInfo(conn2);

    engine1->loadRemoteConnInfo(agent2, conn2);
    engine2->loadRemoteConnInfo(agent1, conn1);

    std::cout << "Establishing connections...\n";
    engine1->connect(agent2);
    engine2->connect(agent1);

    // Wait for async connection establishment to complete
    // The CM thread handles connection progress
    sleep(2);
    std::cout << "Connections established\n\n";

    // Load remote metadata
    nixlBackendMD *recv_rmd = nullptr;

    loadRemote(engine1, 1, agent2, VRAM_SEG, recv_buf, TOTAL_SIZE, recv_md, recv_rmd);

    // Create descriptor lists with different offsets
    nixl_meta_dlist_t src_descs(VRAM_SEG);
    nixl_meta_dlist_t dst_descs(VRAM_SEG);

    populateDescs(src_descs, 0, send_buf, DESC_COUNT, DESC_SIZE, send_md);
    populateDescs(dst_descs, 1, recv_buf, DESC_COUNT, DESC_SIZE, recv_rmd);

    std::cout << "Created " << src_descs.descCount() << " source descriptors\n";
    std::cout << "Created " << dst_descs.descCount() << " destination descriptors\n\n";

    // Perform transfer
    performTransfer(engine1, engine2, src_descs, dst_descs, send_buf, recv_buf, TOTAL_SIZE, NIXL_WRITE);

    // Verify data correctness for each descriptor's region
    std::cout << "\nData verification:\n";
    bool all_correct = true;

    for (int i = 0; i < DESC_COUNT; i++) {
        size_t offset = i * DESC_SIZE;
        uint8_t expected_pattern = static_cast<uint8_t>(i);
        bool desc_correct = true;

        for (size_t j = 0; j < DESC_SIZE; j++) {
            if (((uint8_t *)recv_buf)[offset + j] != expected_pattern) {
                std::cerr << "  ERROR: Descriptor " << i << " at offset " << offset + j
                          << " has wrong data: expected " << (int)expected_pattern << ", got "
                          << (int)((uint8_t *)recv_buf)[offset + j] << "\n";
                desc_correct = false;
                all_correct = false;
                break; // Only report first mismatch per descriptor
            }
        }

        if (desc_correct) {
            std::cout << "  Descriptor " << i << " (offset " << offset << "): OK (pattern "
                      << (int)expected_pattern << ")\n";
        }
    }

    if (all_correct) {
        std::cout << "\n✓ ALL DESCRIPTORS VERIFIED SUCCESSFULLY\n";
        std::cout << "  Each descriptor transferred data from its correct offset\n";
    } else {
        std::cerr << "\n✗ DATA CORRUPTION DETECTED\n";
        std::cerr << "  Some descriptors received data from wrong offsets\n";
        std::cerr << "  This indicates the descriptor offset bug is present!\n";
        exit(1);
    }

    // Cleanup
    engine1->disconnect(agent2);
    engine2->disconnect(agent1);

    deallocateAndDeregister(engine1, 0, VRAM_SEG, send_buf, send_md);
    deallocateAndDeregister(engine2, 1, VRAM_SEG, recv_buf, recv_md);

    releaseEngine(engine1);
    releaseEngine(engine2);

    std::cout << "\nTest completed successfully!\n";
}

int
main(int argc, char **argv) {
    bool p_thread = false;

    if (argc > 1 && std::string(argv[1]) == "--pthread") { 
        p_thread = true;
    }

    test_multi_descriptor_offsets(p_thread);

    return 0;
}
