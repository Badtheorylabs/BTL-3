#include "llama.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        llama_backend_free();
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 8;
    context_params.n_ubatch = 8;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
