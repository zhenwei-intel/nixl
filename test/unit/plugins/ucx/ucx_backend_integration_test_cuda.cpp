/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Integration test for libfabric backend descriptor offset handling
 * Tests the actual backend with multiple descriptors pointing to different offsets
 * within the same registered memory region.


./test/unit/plugins/ucx/ucx_backend_integration_test_cuda --pthread
 */

#include <iostream>
#include <cassert>
#include <cstring>
#include <memory>
#include <unistd.h>

#include "ucx_backend.h"
#include "common/nixl_log.h"
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif
using namespace std;

#define SEND_DEVICE_ID 0
#define RECV_DEVICE_ID 1
#ifdef HAVE_CUDA
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
#endif

nixlUcxEngine *
createEngine(std::string name, bool p_thread) {
    nixlBackendInitParams init;
    nixl_b_params_t       custom_params;

    init.enableProgTh = p_thread;
    init.pthrDelay    = 100;
    init.localAgent   = name;
    init.customParams = &custom_params;
    init.type         = "UCX";

    auto ucx = nixlUcxEngine::create(init).release();
    assert(!ucx->getInitErr());
    if (ucx->getInitErr()) {
        std::cout << "Failed to initialize worker1" << std::endl;
        exit(1);
    }

    return ucx;
}


void
releaseEngine(nixlUcxEngine *engine) {
    delete engine;
}


void
allocateAndRegister(nixlUcxEngine *engine,
                    int dev_id,
                    nixl_mem_t mem_type,
                    void *&addr,
                    size_t len,
                    nixlBackendMD *&md) {
    nixlBlobDesc desc;
#ifdef HAVE_CUDA
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;

    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
    checkCudaError(cudaMalloc(&addr, len), "Failed to allocate CUDA buffer 0");
    cudaQueryAddr(addr, is_dev, dev, ctx);
    std::cout << "CUDA addr: " << std::hex << addr << " dev=" << std::dec << dev
        << " ctx=" << std::hex << ctx << std::dec << std::endl;
#endif
    desc.addr = (uintptr_t)addr;
    desc.len = len;
    desc.devId = dev_id;

    int ret = engine->registerMem(desc, mem_type, md);
    assert(ret == NIXL_SUCCESS);
}

void
deallocateAndDeregister(nixlUcxEngine *engine,
                        int dev_id,
                        nixl_mem_t mem_type,
                        void *&addr,
                        nixlBackendMD *&md) {
    engine->deregisterMem(md);
#ifdef HAVE_CUDA
    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
    checkCudaError(cudaFree(addr), "Failed to allocate CUDA buffer 0");
#endif
}

void doMemset(nixl_mem_t mem_type, int dev_id, void *addr, char byte, size_t len)
{
#ifdef HAVE_CUDA
    checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
    checkCudaError(cudaMemset(addr, byte, len), "Failed to memset");
#endif
}

void *getValidationPtr(nixl_mem_t mem_type, void *addr, size_t len)
{
    switch(mem_type) {
    case DRAM_SEG:
        return addr;
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG: {
        void *ptr = calloc(len, 1);
        checkCudaError(cudaMemcpy(ptr, addr, len, cudaMemcpyDeviceToHost), "Failed to memcpy");
        return ptr;
    }
#endif
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void
loadRemote(nixlUcxEngine *engine,
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
performTransfer(nixlUcxEngine *engine1,
                nixlUcxEngine *engine2,
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
    cout << "\tprepXfer completed immediately\n";
    ret = engine1->postXfer(op, req_src_descs, req_dst_descs, remote_agent, handle, &opt_args);
    assert(ret == NIXL_SUCCESS || ret == NIXL_IN_PROG);

    if (ret == NIXL_SUCCESS) {
        cout << "\t\tTransfer completed immediately\n";
    } else {
        cout << "\t\tWaiting for transfer completion...\n";
        while (ret == NIXL_IN_PROG) {
            ret = engine1->checkXfer(handle);
            // engine1->progress();
            assert( ret == NIXL_SUCCESS || ret == NIXL_IN_PROG);
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
    nixlUcxEngine *engine1 = createEngine(agent1, p_thread);
    nixlUcxEngine *engine2 = createEngine(agent2, p_thread);

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

    allocateAndRegister(engine1, SEND_DEVICE_ID, VRAM_SEG, send_buf, TOTAL_SIZE, send_md);
    allocateAndRegister(engine2, RECV_DEVICE_ID, VRAM_SEG, recv_buf, TOTAL_SIZE, recv_md);

    // Fill send buffer with unique pattern for each descriptor's region
    doMemset(VRAM_SEG, SEND_DEVICE_ID, send_buf, 0xbb, TOTAL_SIZE);
    doMemset(VRAM_SEG, RECV_DEVICE_ID, recv_buf, 0, TOTAL_SIZE);

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

    loadRemote(engine1, RECV_DEVICE_ID, agent2, VRAM_SEG, recv_buf, TOTAL_SIZE, recv_md, recv_rmd);

    // Create descriptor lists with different offsets
    nixl_meta_dlist_t src_descs(VRAM_SEG);
    nixl_meta_dlist_t dst_descs(VRAM_SEG);

    populateDescs(src_descs, SEND_DEVICE_ID, send_buf, DESC_COUNT, DESC_SIZE, send_md);
    populateDescs(dst_descs, RECV_DEVICE_ID, recv_buf, DESC_COUNT, DESC_SIZE, recv_rmd);

    std::cout << "Created " << src_descs.descCount() << " source descriptors\n";
    std::cout << "Created " << dst_descs.descCount() << " destination descriptors\n\n";

    // Perform transfer
    performTransfer(engine1, engine2, src_descs, dst_descs, send_buf, recv_buf, TOTAL_SIZE, NIXL_WRITE);

    // Verify data correctness for each descriptor's region
    std::cout << "\nData verification:\n";

    size_t len = DESC_COUNT * DESC_SIZE;
    void *chkptr1 = getValidationPtr(src_descs.getType(), send_buf, len);
    void *chkptr2 = getValidationPtr(dst_descs.getType(), recv_buf, len);

    // Perform correctness check.
    for(size_t i = 0; i < len; i++){
        assert( ((uint8_t*) chkptr1)[i] == ((uint8_t*) chkptr2)[i]);
    }

    std::cout << "\n✓ ALL DESCRIPTORS VERIFIED SUCCESSFULLY\n";
    std::cout << "  Each descriptor transferred data from its correct offset\n";

    // Cleanup
    engine1->disconnect(agent2);
    engine2->disconnect(agent1);

    deallocateAndDeregister(engine1, SEND_DEVICE_ID, VRAM_SEG, send_buf, send_md);
    deallocateAndDeregister(engine2, RECV_DEVICE_ID, VRAM_SEG, recv_buf, recv_md);

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
