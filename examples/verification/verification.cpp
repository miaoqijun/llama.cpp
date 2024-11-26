#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-i input_len] [-k top_k] [-ngl n_gpu_layers] prompt\n", argv[0]);
    printf("\n");
}

static llama_token llama_sampler_verify(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx, llama_token check_id) {
    const auto * logits = llama_get_logits_ith(ctx, idx);

    const int n_vocab = llama_n_vocab(llama_get_model(ctx));

    // TODO: do not allocate each time
    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    llama_sampler_apply(smpl, &cur_p);

    for (int i = 0; i < (int)cur_p.size; i++) {
        if (check_id == cur_p.data[i].id)
            return check_id;
    }

    return cur_p.data[0].id;
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt;
    // number of layers to offload to the GPU
    int ngl = 99;
    // len of input
    int input_len = 1;
    // sampler top-k
    int top_k = 1;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-i") == 0) {
                if (i + 1 < argc) {
                    try {
                        input_len = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-k") == 0) {
                if (i + 1 < argc) {
                    try {
                        top_k = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }

        if(i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
        else {
            print_usage(argc, argv);
            return 1;
        }
    }

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(model, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(model, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    // verification need to store all logits
    std::vector<int8_t> logits(prompt_tokens.size(), true);
    batch.logits = logits.data();

    const auto t_main_start = ggml_time_us();
    int accepted_len;

    // evaluate the current batch with the transformer model
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        return 1;
    }

    // verification
    for (accepted_len = input_len; accepted_len < n_prompt - 1; accepted_len++) {
        llama_token id = llama_sampler_verify(smpl, ctx, accepted_len, prompt_tokens[accepted_len + 1]);
        if(id != prompt_tokens[accepted_len + 1]) {
            char buf[128];
            int n = llama_token_to_piece(model, id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string expected(buf, n);
            n = llama_token_to_piece(model, prompt_tokens[accepted_len + 1], buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string now(buf, n);
            printf("the %d-th token%s is out of top-%d, recommended token: %s\n", accepted_len + 1, now.c_str(), top_k, expected.c_str());
            break;
        }
    }


    const auto t_main_end = ggml_time_us();
    fprintf(stderr, "%s: accept_len=%d, prompt_len=%d, time: %.2f s\n", __func__, accepted_len, n_prompt - 1, (t_main_end - t_main_start) / 1000000.0f);

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
