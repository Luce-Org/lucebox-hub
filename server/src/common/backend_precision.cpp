#include "backend_precision.h"

#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "gpu_runtime_compat.h"
#endif

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dflash::common {
namespace {

std::string backend_device_description(ggml_backend_t backend) {
    if (!backend) {
        return {};
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return {};
    }
    const char * desc = ggml_backend_dev_description(dev);
    if (desc && desc[0]) {
        return desc;
    }
    return {};
}

std::string backend_device_logical_name(ggml_backend_t backend) {
    if (!backend) {
        return {};
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return {};
    }
    const char * name = ggml_backend_dev_name(dev);
    return name ? std::string(name) : std::string{};
}

std::string backend_name(ggml_backend_t backend) {
    if (!backend) {
        return {};
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return {};
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name ? std::string(name) : std::string{};
}

int parse_backend_device_id(const std::string & logical_name) {
    if (logical_name.empty()) return -1;
    size_t end = logical_name.size();
    while (end > 0 && std::isspace((unsigned char)logical_name[end - 1])) {
        --end;
    }
    size_t begin = end;
    while (begin > 0 && std::isdigit((unsigned char)logical_name[begin - 1])) {
        --begin;
    }
    if (begin == end) return -1;
    return std::atoi(logical_name.substr(begin, end - begin).c_str());
}

int current_device_id() {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
        return -1;
    }
    return device;
#else
    return -1;
#endif
}

int device_props_for(int device,
                     std::string * device_name,
                     std::string * arch_name) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (device < 0) {
        return 0;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return 0;
    }
    if (device_name && prop.name[0]) {
        *device_name = prop.name;
    }
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (arch_name && prop.gcnArchName[0]) {
        *arch_name = prop.gcnArchName;
    }
#endif
    return prop.major * 10 + prop.minor;
#else
    (void)device;
    (void)device_name;
    (void)arch_name;
    return 0;
#endif
}

bool arch_starts_with(const std::string & arch, const char * prefix) {
    return arch.compare(0, std::strlen(prefix), prefix) == 0;
}

bool parse_precision_override(const char * env_name, ggml_type & out) {
    if (!env_name || !env_name[0]) return false;
    const char * s = std::getenv(env_name);
    if (!s || !s[0]) return false;
    if (std::strcmp(s, "f32") == 0 || std::strcmp(s, "F32") == 0) {
        out = GGML_TYPE_F32;
        return true;
    }
    if (std::strcmp(s, "f16") == 0 || std::strcmp(s, "F16") == 0) {
        out = GGML_TYPE_F16;
        return true;
    }
    if (std::strcmp(s, "bf16") == 0 || std::strcmp(s, "BF16") == 0) {
        out = GGML_TYPE_BF16;
        return true;
    }
    std::fprintf(stderr, "[precision] ignoring unsupported %s=%s\n", env_name, s);
    return false;
}

void fill_policy_device_info(ggml_backend_t backend,
                             std::string & backend_name_out,
                             std::string & device_name_out,
                             std::string & runtime_arch_out,
                             int & device_id_out,
                             int & cuda_sm_out) {
    backend_name_out = backend_name(backend);
    const std::string logical_name = backend_device_logical_name(backend);
    device_name_out = backend_device_description(backend);
    device_id_out = parse_backend_device_id(logical_name);
    if (device_id_out < 0) {
        device_id_out = current_device_id();
    }
    cuda_sm_out = device_props_for(
        device_id_out,
        device_name_out.empty() ? &device_name_out : nullptr,
        &runtime_arch_out);
    if (backend_name_out.empty()) backend_name_out = "unknown";
    if (device_name_out.empty()) device_name_out = "unknown";
}

} // namespace

const char * backend_precision_type_name(ggml_type type) {
    return ggml_type_name(type);
}

ggml_type select_cuda_backend_precision_type_for_sm(int sm) {
    if (sm >= 80) return GGML_TYPE_BF16;
    if (sm >= 70 || sm == 60) return GGML_TYPE_F16;
    return GGML_TYPE_F32;
}

ggml_type select_hip_activation_precision_type_for_arch(const std::string & arch) {
    if (arch.empty()) return GGML_TYPE_F32;
    if (arch_starts_with(arch, "gfx90a") ||
        arch_starts_with(arch, "gfx94")  ||
        arch_starts_with(arch, "gfx95")  ||
        arch_starts_with(arch, "gfx11")  ||
        arch_starts_with(arch, "gfx12")) {
        return GGML_TYPE_BF16;
    }
    if (arch_starts_with(arch, "gfx9") ||
        arch_starts_with(arch, "gfx10")) {
        return GGML_TYPE_F16;
    }
    return GGML_TYPE_F32;
}

ggml_type combine_activation_precision_types(ggml_type a, ggml_type b) {
    if (a == GGML_TYPE_F32 || b == GGML_TYPE_F32) return GGML_TYPE_F32;
    if (a == GGML_TYPE_F16 || b == GGML_TYPE_F16) return GGML_TYPE_F16;
    if (a == GGML_TYPE_BF16 && b == GGML_TYPE_BF16) return GGML_TYPE_BF16;
    return GGML_TYPE_F32;
}

BackendPrecisionPolicy select_drafter_precision_policy(ggml_backend_t backend) {
    BackendPrecisionPolicy policy;
    fill_policy_device_info(backend, policy.backend_name, policy.device_name,
                            policy.runtime_arch, policy.device_id,
                            policy.cuda_sm);

#if defined(DFLASH27B_BACKEND_CUDA)
    const ggml_type type = select_cuda_backend_precision_type_for_sm(policy.cuda_sm);
    if (type == GGML_TYPE_BF16) {
        policy.weight_type  = GGML_TYPE_BF16;
        policy.compute_type = GGML_TYPE_BF16;
        policy.reason       = "CUDA sm80+ BF16 tensor-core path";
    } else if (type == GGML_TYPE_F16 && policy.cuda_sm >= 70) {
        policy.weight_type  = GGML_TYPE_F16;
        policy.compute_type = GGML_TYPE_F16;
        policy.reason       = "CUDA sm70-sm79 F16 tensor-core path";
    } else if (type == GGML_TYPE_F16) {
        policy.weight_type  = GGML_TYPE_F16;
        policy.compute_type = GGML_TYPE_F16;
        policy.reason       = "CUDA sm60 GP100 F16 path";
    } else {
        policy.weight_type  = GGML_TYPE_F32;
        policy.compute_type = GGML_TYPE_F32;
        policy.reason       = "CUDA legacy compatibility fallback without useful F16/BF16 acceleration";
    }
#elif defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    policy.weight_type  = GGML_TYPE_BF16;
    policy.compute_type = GGML_TYPE_BF16;
    policy.reason       = "HIP ROCm/ggml BF16-compatible path";
#else
    policy.weight_type  = GGML_TYPE_F32;
    policy.compute_type = GGML_TYPE_F32;
    policy.reason       = "portable non-GPU fallback";
#endif

    return policy;
}

BackendActivationPolicy select_common_activation_precision_policy(
        const std::vector<ggml_backend_t> & backends,
        bool force_f32,
        const char * override_env) {
    if (backends.empty()) {
        return select_activation_precision_policy(nullptr, force_f32, override_env);
    }

    BackendActivationPolicy policy =
        select_activation_precision_policy(backends.front(), force_f32, override_env);
    ggml_type common_type = policy.activation_type;
    bool mixed = false;
    for (size_t i = 1; i < backends.size(); ++i) {
        const BackendActivationPolicy shard_policy =
            select_activation_precision_policy(backends[i], force_f32, override_env);
        if (shard_policy.activation_type != common_type) {
            mixed = true;
        }
        const ggml_type combined =
            combine_activation_precision_types(common_type, shard_policy.activation_type);
        if (combined != common_type) {
            mixed = true;
        }
        common_type = combined;
    }
    if (mixed) {
        policy.activation_type = common_type;
        policy.backend_name = "mixed";
        policy.device_name = "mixed";
        policy.runtime_arch = "mixed";
        policy.device_id = -1;
        policy.cuda_sm = 0;
        policy.reason = "common shard-compatible activation path";
    }
    return policy;
}

BackendActivationPolicy select_activation_precision_policy(
        ggml_backend_t backend,
        bool force_f32,
        const char * override_env) {
    BackendActivationPolicy policy;
    fill_policy_device_info(backend, policy.backend_name, policy.device_name,
                            policy.runtime_arch, policy.device_id,
                            policy.cuda_sm);

    ggml_type override_type = GGML_TYPE_F32;
    if (!force_f32 && parse_precision_override(override_env, override_type)) {
        policy.activation_type = override_type;
        policy.reason = std::string(override_env) + " override";
        return policy;
    }
    if (force_f32) {
        policy.activation_type = GGML_TYPE_F32;
        policy.reason = "F32 required by capture/IPC feature boundary";
        return policy;
    }

#if defined(DFLASH27B_BACKEND_CUDA)
    policy.activation_type = select_cuda_backend_precision_type_for_sm(policy.cuda_sm);
    if (policy.activation_type == GGML_TYPE_BF16) {
        policy.reason = "CUDA sm80+ BF16 activation path";
    } else if (policy.activation_type == GGML_TYPE_F16 && policy.cuda_sm >= 70) {
        policy.reason = "CUDA sm70-sm79 F16 activation path";
    } else if (policy.activation_type == GGML_TYPE_F16) {
        policy.reason = "CUDA sm60 GP100 F16 activation path";
    } else {
        policy.reason = "CUDA legacy F32 activation fallback";
    }
#elif defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    policy.activation_type =
        select_hip_activation_precision_type_for_arch(policy.runtime_arch);
    if (policy.activation_type == GGML_TYPE_BF16) {
        policy.reason = "HIP native BF16 activation path";
    } else if (policy.activation_type == GGML_TYPE_F16) {
        policy.reason = "HIP FP16 activation path";
    } else {
        policy.reason = "HIP legacy F32 activation fallback";
    }
#else
    policy.activation_type = GGML_TYPE_F32;
    policy.reason = "portable F32 activation fallback";
#endif
    return policy;
}

} // namespace dflash::common
