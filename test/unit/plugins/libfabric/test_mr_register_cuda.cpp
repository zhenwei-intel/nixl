#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <iostream>
#include "libfabric_common.h"
#include "libfabric_rail.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif
using namespace std;


void* allocate_device_memory(size_t len, int dev_id) {
    cudaSetDevice(dev_id);
    void* device_ptr;
    cudaError_t status = cudaMalloc(&device_ptr, len);
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        return nullptr;
    }
    return device_ptr;
}


// Provider configuration structure
struct ProviderConfigTest {
    std::string name;
    uint64_t caps;
    uint64_t mode;
    uint64_t mr_mode;
    fi_resource_mgmt resource_mgmt;
    fi_threading threading;
};

static const ProviderConfigTest PROVIDER_CONFIGS[] = {
    {
        "efa",
        FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
        FI_CONTEXT | FI_CONTEXT2,
        0,  // let provider choose
        FI_RM_UNSPEC,
        FI_THREAD_SAFE
    },
    {
        "verbs",  // Matches both "verbs" and "verbs;ofi_rxm"
        FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_MULTI_RECV | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_HMEM,
        0,  // no mode flags required
        FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM,
        FI_RM_ENABLED,
        FI_THREAD_SAFE
    },
    {
        "tcp",
        FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
        FI_CONTEXT | FI_CONTEXT2,
        0,  // basic MR mode, overridden in rail.cpp
        FI_RM_UNSPEC,
        FI_THREAD_UNSPEC
    },
    {
        "sockets",
        FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
        0,
        0,  // let provider choose
        FI_RM_UNSPEC,
        FI_THREAD_UNSPEC  // default threading
    },
    {
        "shm",
        FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_MULTI_RECV | FI_LOCAL_COMM | FI_RMA_EVENT | FI_SOURCE | FI_DIRECTED_RECV | FI_HMEM,
        0,
        FI_MR_VIRT_ADDR| FI_MR_HMEM,
        FI_RM_ENABLED,
        FI_THREAD_SAFE  // default threading
    }
};

static const size_t NUM_PROVIDER_CONFIGS = sizeof(PROVIDER_CONFIGS) / sizeof(PROVIDER_CONFIGS[0]);


void configureHintsForProvider(struct fi_info* hints, const std::string& provider_name) {
    const ProviderConfigTest* config = nullptr;

    // Find matching config
    // Match order: 1) exact match, 2) prefix match for composite providers (e.g., "verbs;ofi_rxm")
    for (size_t i = 0; i < NUM_PROVIDER_CONFIGS; ++i) {
        const std::string& config_name = PROVIDER_CONFIGS[i].name;

        // Exact match
        if (provider_name == config_name) {
            config = &PROVIDER_CONFIGS[i];
            break;
        }

        // Composite provider match (e.g., "verbs;ofi_rxm" matches "verbs")
        // Check if provider_name starts with config_name followed by ";"
        if (provider_name.rfind(config_name + ";", 0) == 0) {
            config = &PROVIDER_CONFIGS[i];
            break;
        }
    }

    if (!config) {
        // Default configuration
        std::cout << "No specific config for provider '" << provider_name << "', using defaults" << std::endl;
        hints->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM;
        hints->mode = 0;
        hints->ep_attr->type = FI_EP_RDM;
        return;
    }

    // Apply provider-specific configuration
    hints->caps = config->caps;
    hints->mode = config->mode;
    hints->ep_attr->type = FI_EP_RDM;

    if (config->resource_mgmt != FI_RM_UNSPEC) {
        hints->domain_attr->resource_mgmt = config->resource_mgmt;
    }

    if (config->mr_mode != 0) {
        hints->domain_attr->mr_mode = config->mr_mode;
    }

    if (config->threading != FI_THREAD_UNSPEC) {
        hints->domain_attr->threading = config->threading;
    }
}

int init_provider(char* provider_name) {

        // Get fabric device info with PCIe addresses from libfabric
    struct fi_info *hints, *info;

    hints = fi_allocinfo();
    if (!hints) {
        std::cerr << "Failed to alloc fi_info" << std::endl;
        return -1;
    }

    // Configure hints based on provider
    configureHintsForProvider(hints, provider_name);

    // Override mr_mode for TCP/sockets (they don't support advanced MR features)
    if (provider_name == "tcp" || provider_name == "sockets") {
        hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ALLOCATED;
        hints->domain_attr->mr_key_size = 0; // Let provider decide
    } else {
        // Add HMEM support for other providers (EFA, verbs)
        if (hints->domain_attr->mr_mode != 0) {
            hints->domain_attr->mr_mode |= FI_MR_HMEM;
        } else {
            hints->domain_attr->mr_mode =
                FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
        }
        hints->domain_attr->mr_key_size = 2;
    }

    std::string device_name = "shm";

    hints->domain_attr->name = strdup(device_name.c_str());

    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);

    std::cout << "verbs provider initialized successfully." << std::endl;

    // Test rail
    // ret = initializeXPU();
    // if (ret < 0) return -1;

    size_t length = 2048;
    auto xpu_device_ptr = allocate_device_memory(length, 0);

    auto fabric_rail = nixlLibfabricRail("shm", "shm", static_cast<uint16_t>(0));

    struct fid_domain *domain = nullptr;
    struct fid_fabric *fabric = nullptr;

    // Create fabric for this rail
    ret = fi_fabric(info->fabric_attr, &fabric, NULL);
    if (ret) {
        std::cerr << "fi_fabric failed with " << ret << std::endl;
        return -1;
    }
    std::cout << "fabric_attr->name " << info->fabric_attr->name << std::endl;
    // Create domain for this rail
    ret = fi_domain(fabric, info, &domain, NULL);
    if (ret) {
        std::cerr << "fi_domain failed with " << ret << std::endl;
        return -1;
    }

    // register memory
    struct fid_mr *mr;
    struct fid_mr *chunk_mr;

    // Use fi_mr_regattr for HMEM device memory registration
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {};

    iov.iov_base = xpu_device_ptr;
    iov.iov_len = length;

    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    // get access from  rail
    mr_attr.access = fabric_rail.getMemoryRegistrationAccessFlags();

    mr_attr.iface = FI_HMEM_ZE;
    mr_attr.device.ze = 0;  // device_id

    size_t chunk_size =8388608;
    void* chunk_buffer = malloc(chunk_size);
    ret = fi_mr_reg(domain, chunk_buffer, chunk_size, FI_SEND | FI_RECV, 0, 0, 0, &chunk_mr, NULL);
    if (ret) {
        std::cout << "fi_mr_reg Failed \n" << std::endl;
    } else {
        std::cout << "fi_mr_reg Passed \n" << std::endl;
    }

    ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);

     if (ret) {
         std::cout << "fi_mr_regattr Failed \n" << std::endl;
     } else {
         std::cout << "fi_mr_regattr Passed \n" << std::endl;
     }

    fi_freeinfo(info);
    fi_freeinfo(hints);
    return 0;
}

int main(int argc, char **argv) {
    char *provider_name = NULL;
        provider_name = argv[1];
        std::cout << "Input provider as " << provider_name << std::endl;
//    auto network_device = LibfabricUtils::getAvailableNetworkDevices();
//    return 0;
    return init_provider(provider_name);
}


