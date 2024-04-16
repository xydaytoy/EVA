#模型保存路径
models = dict()
#模块名称
module_names = dict()
#词表大小
vocab_sizes = dict()
#cos矩阵存储的路径
matrix_paths = dict()



#7b models
#llama2-7b-chat
models["llama2"] = "/data/xyyf/models/llama2-7b-chat-hf"
module_names["llama2"] = "lm_head.weight"#[32000, 4096]
vocab_sizes["llama2"] = 32000
matrix_paths["llama2"] = {
    "chatglm2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/chatglm2-llama2_top-10.npz",
    },
    "baichuan2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/baichuan2-llama2_top-10.npz",
    },
    "internlm":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/internlm-llama2_top-10.npz",
    },
}


#baichuan2-7b-chat
models["baichuan2"] = "/data/xyyf/models/baichuan2-7b-chat"
module_names["baichuan2"] = "lm_head.weight"#[125696, 4096]
vocab_sizes["baichuan2"] = 125696
matrix_paths["baichuan2"] = {
    "chatglm2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/chatglm2-baichuan2_top-10.npz",
    },
    "llama2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/llama2-baichuan2_top-10.npz",
    },
    "internlm":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/internlm-baichuan2_top-10.npz",
    },
}


#chatglm2-6b
models["chatglm2"] = "/data/xyyf/models/chatglm2-6b"
module_names["chatglm2"] = "transformer.output_layer.weight"#[65024, 4096]
vocab_sizes["chatglm2"] = 65024
matrix_paths["chatglm2"] = {
    "baichuan2":{ 
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/baichuan2-chatglm2_top-10.npz"    
    },
    "internlm":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/internlm-chatglm2_top-10.npz",
    },
    "llama2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/llama2-chatglm2_top-10.npz",
    },
}

#internlm-chat-v1.1
models["internlm"] = "/data/xyyf/models/internlm-chat-7b-v1_1"
module_names["internlm"] = "lm_head.weight"#[103168, 4096]
vocab_sizes["internlm"] = 103168
matrix_paths["internlm"] = {
    "baichuan2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/baichuan2-internlm_top-10.npz",
    },
    "llama2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/llama2-internlm_top-10.npz",
    },
    "chatglm2":{
        "filter":"/data/xyyf/EVA/sparse_matrix_filter/chatglm2-internlm_top-10.npz",
    },
}


inst_cot_prompts = {
    "llama2-13b": "[INST] <<SYS>>\nPlease solve the following math problem:\n<</SYS>>\n\nQuestion: {instruction} [/INST] Answer: Let's think step by step.",
    "baichuan2-13b": "<reserved_106>Please solve the following math problem:\nQuestion: {instruction}<reserved_107>Answer: Let's think step by step.",
    "tigerbot-13b": "### Instruction:\nPlease solve the following math problem:\nQuestion: {instruction}\n### Response:\nAnswer: Let's think step by step.",
    "vicuna-13b": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Please solve the following math problem:\nQuestion:{instruction}\nASSISTANT: Answer: Let's think step by step.",
    "wizard-13b": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Please solve the following math problem:\nQuestion:{instruction}\nASSISTANT: Answer: Let's think step by step.",
    "chinese-alpaca-13b": "[INST] <<SYS>>\nPlease solve the following math problem:\n<</SYS>>\n\nQuestion: {instruction} [/INST] Answer: Let's think step by step.",
    "llama2": "[INST] <<SYS>>\nPlease solve the following math problem:\n<</SYS>>\n\nQuestion: {instruction} [/INST] Answer: Let's think step by step.",
    "chatglm2": "[Round 1]\nPlease solve the following math problem:\nQuestion: {instruction}\nAnswer: Let's think step by step.",
    "baichuan2": "<reserved_106>Please solve the following math problem:\nQuestion: {instruction}<reserved_107>Answer: Let's think step by step.",
    "internlm": "<|User|>:Please solve the following math problem:\nQuestion: {instruction}<eoh>\n<|Bot|>:Answer: Let's think step by step.",
    "tigerbot": "### Instruction:\nPlease solve the following math problem:\nQuestion: {instruction}\n### Response:\nAnswer: Let's think step by step.",
    "vicuna": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Please solve the following math problem:\nQuestion:{instruction}\nASSISTANT: Answer: Let's think step by step.",
    "chinese-alpaca": "[INST] <<SYS>>\nPlease solve the following math problem:\n<</SYS>>\n\nQuestion: {instruction} [/INST] Answer: Let's think step by step.",
}


prompt_schemas = {
    "chinese-alpaca-13b": {
        "instruction_prefix": "[INST] <<SYS>>\n",
        "instruction_suffix": "<</SYS>>\n\n",
        "input_prefix": "",
        "input_suffix": " [/INST] "
    },

    "llama2-13b": {
        "instruction_prefix": "[INST] <<SYS>>\n",
        "instruction_suffix": "<</SYS>>\n\n",
        "input_prefix": "",
        "input_suffix": " [/INST] "
    },

    "baichuan2-13b":{
        "instruction_prefix": "<reserved_106>",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "\n<reserved_107>"
    },

    "tigerbot-13b":{
        "instruction_prefix": "\n\n### Instruction:\n",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "\n\n### Response:\n"
    },

    "vicuna-13b": {
        "instruction_prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "\nASSISTANT: "
    },

    "wizard-13b": {
        "instruction_prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "\nASSISTANT: "
    },
    
    "llama2": {
        "instruction_prefix": "[INST] <<SYS>>\n",
        "instruction_suffix": "<</SYS>>\n\n",
        "input_prefix": "",
        "input_suffix": " [/INST] "
    },

    "chatglm2":{
        "instruction_prefix": "[Round 1]\n\n",
        "instruction_suffix": "",
        "input_prefix": "问：",
        "input_suffix": "\n\n答："
    },

    "baichuan2":{
        "instruction_prefix": "<reserved_106>",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "\n<reserved_107>"
    },

    "internlm":{
        "instruction_prefix": "<|User|>:",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "<eoh>\n<|Bot|>:"
    },

    "tigerbot":{
        "instruction_prefix": "\n\n### Instruction:\n",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "\n\n### Response:\n"
    },

    "vicuna": {
        "instruction_prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
        "instruction_suffix": "",
        "input_prefix": "",
        "input_suffix": "\nASSISTANT: "
    },
    
    "chinese-alpaca": {
        "instruction_prefix": "[INST] <<SYS>>\n",
        "instruction_suffix": "<</SYS>>\n\n",
        "input_prefix": "",
        "input_suffix": " [/INST] "
    }
}