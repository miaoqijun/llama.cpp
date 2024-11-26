#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <future>

#define MIN_VERIFY_BATCH_SIZE (64)

struct parameters {
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
    // verify threads
    int n_thread = 1;
};

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-i input_len] [-k top_k] [-t verify_thread] [-ngl n_gpu_layers] \
    -f file contatining prompt or just input prompt here\n", argv[0]);
    printf("\n");
}

static int read_file(char * path, std::string &content)
{
    std::ifstream file(path);
    if (!file.is_open())
        return 1;

    std::string line;
    while (std::getline(file, line))
        content += line + "\n";
    file.close();

    return 0;
}

static int parse_args(int argc, char ** argv, parameters & params)
{
    int i = 1;
    for (; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            params.model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                try {
                    params.input_len = std::stoi(argv[++i]);
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
                    params.top_k = std::stoi(argv[++i]);
                } catch (...) {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                try {
                    params.n_thread = std::stoi(argv[++i]);
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
                    params.ngl = std::stoi(argv[++i]);
                } catch (...) {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-f") == 0) {
            if (i + 1 < argc) {
                if(read_file(argv[++i], params.prompt)) {
                    printf("\nerror: can not load prompt from %s\n", argv[i]);
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

    if (params.model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    if (params.prompt.empty()) {
        if(i < argc) {
            params.prompt = argv[i++];
            for (; i < argc; i++) {
                params.prompt += " ";
                params.prompt += argv[i];
            }
        }      
        else {
            print_usage(argc, argv);
            return 1;
        }  
    }

    return 0;
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

static void verify(int token_start, int token_end, struct llama_sampler * smpl, struct llama_context * ctx, std::vector<llama_token> & prompt_tokens, std::promise<int> prom)
{
    int pos;
    for (pos = token_start; pos < token_end; pos++) {
        if(llama_sampler_verify(smpl, ctx, pos - 1, prompt_tokens[pos]) != prompt_tokens[pos])
            break;
    }
    prom.set_value(pos);
}

int main(int argc, char ** argv) {
    struct parameters params;

    // parse command line arguments

    if (parse_args(argc, argv, params))
        return 1;
    
    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.ngl;

    llama_model * model = llama_load_model_from_file(params.model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(model, params.prompt.c_str(), params.prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(model, params.prompt.c_str(), params.prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
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

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    // verification need to store all logits
    std::vector<int8_t> logits(prompt_tokens.size(), true);
    batch.logits = logits.data();

    const auto t_main_start = ggml_time_us();

    // evaluate the current batch with the transformer model
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        return 1;
    }

    const auto t_main_decoded = ggml_time_us();

    // verification
    int token_per_thread = (n_prompt - params.input_len - 1 + params.n_thread - 1) / params.n_thread;
    token_per_thread = token_per_thread > MIN_VERIFY_BATCH_SIZE ? token_per_thread : MIN_VERIFY_BATCH_SIZE;
    printf("token_per_thread: %d\n", token_per_thread);
    std::vector<std::thread> threads;
    std::vector<std::future<int>> futures;

    for (int token_start = params.input_len + 1; token_start < n_prompt; token_start += token_per_thread) {
        std::promise<int> prom;
        futures.push_back(prom.get_future());

        int token_end = token_start + token_per_thread < n_prompt ? token_start + token_per_thread : n_prompt;
        threads.emplace_back(verify, token_start, token_end, smpl, ctx, std::ref(prompt_tokens), std::move(prom));
    }
    int i = 0;
    for (int token_start = params.input_len + 1; token_start < n_prompt; token_start += token_per_thread) {
        int result = futures[i++].get();
        int token_end = token_start + token_per_thread < n_prompt ? token_start + token_per_thread : n_prompt;
        if (result < token_end) {
            char buf[128];
            int n = llama_token_to_piece(model, prompt_tokens[result], buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("the %d-th token %s is out of top-%d\n", result, s.c_str(), params.top_k);
            break;
        }
    }
    for (auto& t : threads) {
        t.join();
    }

    const auto t_main_end = ggml_time_us();
    fprintf(stderr, "\n%s: decoding time: %.2f s, verification time: %.2f s\n", __func__, (t_main_decoded - t_main_start) / 1000000.0f, (t_main_end - t_main_decoded) / 1000000.0f);

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
