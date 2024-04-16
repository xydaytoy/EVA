import argparse
import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
import scipy
import scipy.sparse as sp
import torch.nn.functional as F
import json
import sys
sys.path.append('/data/xyyf/EVA')
import model_info
from model_info import models, matrix_paths, prompt_schemas, inst_cot_prompts
import task_info
from task_info import save_dirs, max_new_tokens,get_test_df, get_dev_df,clean_answer
from tqdm import tqdm
device0 = 'cuda:0'
device1 = 'cuda:1'

prompt_key_dict = {
    "nq":'qText',
    "triviaqa":'qText',
    "addsub":'input',
    "asdiv":'input',
    "gsm8k":'question',
    "e2e":'concepts',
}

short2lang = {
    'eng': 'English',
    'zho_simpl': 'Chinese',
}

count_num = [0,0,0,0,0,0,0]#统计集成模型的个数

def format_example(src, tgt, s, t=None):
    prompt =  "{}:{}={}:".format(short2lang[src],s,short2lang[tgt])
    if t is not None:
        prompt += "{}\n".format(t)
    return prompt

def gen_prompt(train_df, src, tgt, k=4):
    prompt = "Translate the following sentence from "+short2lang[src]+" to "+short2lang[tgt]+".\n"
    if k == -1:
        k = len(train_df["src"])
    for i in range(k):
        prompt += format_example(src, tgt, train_df["src"][i], train_df["tgt"][i])
    return prompt

def build_inst_prompt(task, model, question):
    if task == "gsm8k" or task == "addsub" or task == "asdiv":
        prompt = inst_cot_prompts[model].format_map({"instruction": question})
        return prompt
    prompt_schema = prompt_schemas[model]
    model_instruction_prefix = prompt_schema["instruction_prefix"]
    model_instruction_suffix = prompt_schema["instruction_suffix"]
    model_input_prefix = prompt_schema["input_prefix"]
    model_input_suffix = prompt_schema["input_suffix"]

    if task == 'nq' or task == 'triviaqa':
        instruction = "Please answer the following question, your answer should be as simple as possible.\n"
        inputs = "Question: " + question
        prompt = model_instruction_prefix + instruction + model_instruction_suffix + \
        model_input_prefix + inputs + model_input_suffix + "Answer:"
    elif task == 'e2e':
        instruction = "Please describe all aspects of the restaurant in one sentence based on the following information.\n"
        inputs = "Information: " + question
        prompt = model_instruction_prefix + instruction + model_instruction_suffix + \
        model_input_prefix + inputs + model_input_suffix + "Restaurant description:"
    return prompt

def topk_filter(logits, top_k=40):
    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits
    
def drop(logit, res_logits, k=5):
    flag = False
    top1_token = torch.argmax(logit, dim=-1).item()
    for res_logit in res_logits:
        topk_tokens = torch.argsort(res_logit, descending=True)[:,:k]
        if top1_token in topk_tokens:
            flag = True
    return flag


@torch.no_grad()
def eval_local(args, model, model_aux_list0, model_aux_list1, tokenizer, tokenizer_aux_list, sparse_matrix_list, test_df, dev_df=None, src=None, tgt=None):
    eos_token_id = model.generation_config.eos_token_id
    predictions = []
    if args.task == 'flores':
        example_prompt = gen_prompt(dev_df, src, tgt, 4)
    for obj in tqdm(test_df):
        if args.task == 'flores':
            prompt = example_prompt + format_example(src, tgt, obj)
        else:
            prompt = build_inst_prompt(args.task, args.model, obj[prompt_key_dict[args.task]])
            prompt_aux_list = []
            for aux_model in args.aux_models:
                prompt_aux = build_inst_prompt(args.task, aux_model, obj[prompt_key_dict[args.task]])
                prompt_aux_list.append(prompt_aux)

        prompt_length = len(prompt)
        num_of_new_tokens = 0

        while num_of_new_tokens <= args.MAX_NEW_TOKEN:
            logits_aux_list = []
            with torch.no_grad():
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device0)
                logits = model(input_ids=input_ids).logits[:, -1, :].to(torch.float32)
                logits = topk_filter(logits, top_k=args.topk)
                logits = F.softmax(logits, dim=-1).to('cpu')
                
                len0 = len(model_aux_list0)
                for aux_idx, aux_item in enumerate(zip(tokenizer_aux_list[:len0], model_aux_list0, sparse_matrix_list[:len0])):
                    tokenizer_aux, model_aux, sparse_matrix = aux_item[0], aux_item[1], aux_item[2]
                    if args.task == 'flores':
                        input_ids_aux = tokenizer_aux(prompt, return_tensors="pt").input_ids.to(device0)
                    else:
                        input_ids_aux = tokenizer_aux(prompt_aux_list[aux_idx], return_tensors="pt").input_ids.to(device0)
                    logits_aux = model_aux(input_ids=input_ids_aux).logits[:,-1,:].to(torch.float32)
                    logits_aux = topk_filter(logits_aux, top_k=args.topk)
                    #scale归一化
                    logits_aux = F.softmax(logits_aux, dim=-1)
                    logits_aux = logits_aux.t()
                    logits_aux = torch.spmm(sparse_matrix.to(device0), logits_aux)
                    logits_aux = logits_aux.t()
                    logits_aux = logits_aux.to('cpu')
                    logits_aux_list.append(logits_aux)
                for aux_idx, aux_item in enumerate(zip(tokenizer_aux_list[len0:], model_aux_list1, sparse_matrix_list[len0:])):
                    tokenizer_aux, model_aux, sparse_matrix = aux_item[0], aux_item[1], aux_item[2]
                    if args.task == 'flores':
                        input_ids_aux = tokenizer_aux(prompt, return_tensors="pt").input_ids.to(device1)
                    else:
                        input_ids_aux = tokenizer_aux(prompt_aux_list[aux_idx], return_tensors="pt").input_ids.to(device1)
                    logits_aux = model_aux(input_ids=input_ids_aux).logits[:,-1,:].to(torch.float32)
                    logits_aux = topk_filter(logits_aux, top_k=args.topk)
                    logits_aux = F.softmax(logits_aux, dim=-1)
                    logits_aux = logits_aux.t()
                    logits_aux = torch.spmm(sparse_matrix.to(device1), logits_aux)
                    logits_aux = logits_aux.t()
                    logits_aux = logits_aux.to('cpu')
                    logits_aux_list.append(logits_aux)

            if args.aux_method == "linear_sum":
                ensemble_logits = (1 - sum(args.aux_lambda)) * logits
                for x, y in zip(args.aux_lambda, logits_aux_list):
                    ensemble_logits += x * y
            elif args.aux_method == "drop":
                tmp_list = logits_aux_list
                tmp_list.append(logits)
                res_list = []
                for idx, val in enumerate(tmp_list):
                    res_logits = tmp_list[:idx]+tmp_list[idx+1:]
                    if drop(val, res_logits, args.drop):
                        res_list.append(val)
                count_num[len(res_list)] += 1
                if len(res_list) == 0:
                    res_list = tmp_list
                tmp_weight = 1/len(res_list)
                for idx, val in enumerate(res_list):
                    if idx == 0:
                        ensemble_logits = tmp_weight * val
                    else:
                        ensemble_logits += tmp_weight * val
            else:
                print("error: wrong method name")
            next_ids = torch.argmax(ensemble_logits, dim=-1)
            if next_ids.item() == eos_token_id:
                break
            else:
                num_of_new_tokens += 1
                next_tokens = tokenizer.convert_ids_to_tokens(next_ids.item())
                next_tokens = tokenizer.convert_tokens_to_string(next_tokens)
                prompt += next_tokens
                if args.task != 'flores':
                    for aux_idx in range(len(prompt_aux_list)):
                        prompt_aux_list[aux_idx] = prompt_aux_list[aux_idx] + next_tokens
        input_ids = input_ids.to(device0)
        pred_text = tokenizer.decode(input_ids[0],skip_special_tokens=True)
        if args.task == 'flores':
            pred_text = pred_text[prompt_length:]
            pred_text = pred_text.split('\n')[0]
            pred_text = pred_text.strip()
            predictions.append(pred_text)
        else:
            pred_text0 = pred_text[:prompt_length]
            pred_text1 = pred_text[prompt_length:]
            predictions.append({"prompt":pred_text0,"pred_all":pred_text1})
    return predictions


def main(args):
    #aux模型
    tokenizer_aux_list = []
    model_aux_list0 = []
    model_aux_list1 = []
    sparse_matrix_list = []
    for id, aux_model in enumerate(args.aux_models):
        model_aux_ckpt = models[aux_model]
        tokenizer_aux = AutoTokenizer.from_pretrained(model_aux_ckpt, use_fast=False, add_bos_token=False, model_max_length=4096,padding_side="left",trust_remote_code=True)
        tokenizer_aux_list.append(tokenizer_aux)
        config_aux = AutoConfig.from_pretrained(model_aux_ckpt, trust_remote_code=True)
        if id > 0:
            model_aux = AutoModelForCausalLM.from_pretrained(model_aux_ckpt, config=config_aux, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).to(device1)
            model_aux_list1.append(model_aux)
        else:
            model_aux = AutoModelForCausalLM.from_pretrained(model_aux_ckpt, config=config_aux, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).to(device0)
            model_aux_list0.append(model_aux)

        #加载相似度矩阵
        matrix_path = matrix_paths[args.model][aux_model][args.matrix_name] #保存cos矩阵的位置
        scipy_matrix = sp.load_npz(matrix_path)
        sparse_matrix = torch.sparse_coo_tensor(scipy_matrix.nonzero(), scipy_matrix.data, scipy_matrix.shape).t().to(device0).to(torch.float32)#[32000,65024]
        sparse_matrix_list.append(sparse_matrix)

    #主模型
    model_ckpt = models[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False,add_bos_token=False, model_max_length=4096,padding_side="left",trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, config=config, torch_dtype=torch.bfloat16,trust_remote_code=True, low_cpu_mem_usage=True).to(device0)

    aux_model_str = "_".join(map(str, args.aux_models))
    if args.aux_method == 'drop':
        drop_str = str(args.drop)
    else:
        drop_str = ""
    if args.task == 'flores':
        mode_str = "{}-{}-{}-4shot-{}-{}-{}{}-top{}-{}".format(args.task, args.src, args.tgt, args.model, aux_model_str, args.aux_method, drop_str, str(args.topk), args.matrix_name)
    else:
        mode_str = "{}-{}-{}-{}{}-top{}-{}".format(args.task, args.model, aux_model_str, args.aux_method, drop_str, str(args.topk), args.matrix_name)
    print("Mode: " + mode_str)#nq-xx-xx-xx-xx-linear_sum-top320-filter-inst

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, mode_str)):
        os.makedirs(os.path.join(args.save_dir, mode_str))

    if args.task == 'flores':
        dev_df = dict()
        src = args.src
        tgt = args.tgt
        data_dir = "/data/xyyf/001-corpus/flores101"
        with open(os.path.join(data_dir, "dev", src + ".dev")) as f:
            dev_df["src"] = f.read().splitlines()[:4]
        with open(os.path.join(data_dir, "devtest", src + ".devtest")) as f:
            test_df = f.read().splitlines()
        with open(os.path.join(data_dir, "dev", tgt + ".dev")) as f:
            dev_df["tgt"] = f.read().splitlines()[:4]
        predictions = eval_local(args, model, model_aux_list0, model_aux_list1, tokenizer, tokenizer_aux_list, sparse_matrix_list, test_df, dev_df, src, tgt)
        pred_file = os.path.join(args.save_dir, mode_str, src+'-'+tgt+'.pred')
        with open(pred_file,"w",encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred+'\n')
    else:
        test_df = get_test_df(args.task)
        predictions = eval_local(args, model, model_aux_list0, model_aux_list1, tokenizer, tokenizer_aux_list, sparse_matrix_list, test_df)
        pred_file = os.path.join(args.save_dir, mode_str, 'pred.json')
        with open(pred_file, "w", encoding='utf-8') as f:
            for pred, obj in zip(predictions,test_df):
                obj["prompt"] = pred["prompt"]
                obj["pred_all"] = pred["pred_all"]
                obj["pred"] = clean_answer(args.task, pred["pred_all"])
            json.dump(test_df, f, indent=4)
    if args.aux_method == 'drop':
        with open(os.path.join(args.save_dir, mode_str, 'count_num.txt'),'w',encoding='utf-8') as f:
            f.write(str(count_num[0])+'\t'+str(count_num[1])+'\t'+str(count_num[2])+'\t'+str(count_num[3])+'\t'+str(count_num[4])+'\t'+str(count_num[5])+'\t'+str(count_num[6])+'\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", type=str, default='nq')
    parser.add_argument("--model", "-m", type=str, default="llama2")
    parser.add_argument("--aux_models", nargs="+", type=str, help="A list of models")
    parser.add_argument("--aux_method", type=str, default="linear_sum")
    parser.add_argument("--matrix_name", type=str, default="filter")
    parser.add_argument("--topk", type=int, default=320, help="A list of lambdas")
    parser.add_argument("--drop", type=int, default=5)
    parser.add_argument("--aux_lambda", nargs="+", type=float, help="A list of lambdas")
    parser.add_argument("--save_dir", type=str, default="/data/xyyf/EVA/ensemble/results")
    parser.add_argument("--src", type=str, default="zho_simpl")
    parser.add_argument("--tgt", type=str, default="eng")
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.task)
    args.MAX_NEW_TOKEN = max_new_tokens[args.task]
    if args.aux_lambda == None:
        if args.aux_method != "drop":
            num_of_models = len(args.aux_models)+1
            args.aux_lambda = [1/num_of_models for _ in range(num_of_models)]
        else:
            args.aux_lambda = None
    main(args)
