#pragma once

#include "ggml.h"

namespace dflash::common {

inline ggml_tensor * graph_tensor_f32(ggml_context * ctx, ggml_tensor * x) {
    if (!x || x->type == GGML_TYPE_F32) {
        return x;
    }
    return ggml_cast(ctx, x, GGML_TYPE_F32);
}

inline ggml_tensor * rms_norm_input_f32(ggml_context * ctx, ggml_tensor * x) {
    return graph_tensor_f32(ctx, x);
}

} // namespace dflash::common
