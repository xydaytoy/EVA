from datasets import load_from_disk
import json
import os
import re


#最大生成长度
max_new_tokens = {
    "nq": 32,
    "triviaqa": 32,
    "e2e":64,
    "addsub":256,
    "asdiv":256,
    "gsm8k":256,
    "flores":128,
}


#数据集加载
def get_test_df(task):
    if task == 'nq':
        test_data = load_from_disk("/data/xyyf/102-vocab/dataset/nq_open/validation")
        test_df = []
        for line in test_data:
            test_df.append({"qText":line['question'],"answers":line['answer']})
    elif task == "triviaqa":
        test_data = load_from_disk("/data/xyyf/102-vocab/dataset/trivia_qa/rc")['validation']
        test_df = []
        for line in test_data:
            test_df.append({"qText":line['question'],"answers":line['answer']['aliases']})
    elif task == "addsub":
        test_data = load_from_disk("/data/xyyf/102-vocab/dataset/allenai/lila")['test']
        test_data = test_data.filter(lambda example: example['dataset'] == 'addsub.json')
        test_df = []
        for line in test_data:
            test_df.append(line)
    elif task == "asdiv":
        test_data = load_from_disk("/data/xyyf/102-vocab/dataset/allenai/lila")['test']
        test_data = test_data.filter(lambda example: example['dataset'] == 'asdiv.json')
        test_df = []
        for line in test_data:
            test_df.append(line)
    elif task == 'gsm8k':
        with open(os.path.join("/data/xyyf/001-corpus/gsm8k/grade-school-math-master/grade_school_math/data/test.jsonl"), "r", encoding="utf-8") as f:
            test_df = []
            for line in f:
                test_df.append(json.loads(line))
    elif task == 'e2e':
        test_data = load_from_disk("/data/xyyf/102-vocab/dataset/e2e_nlg")['test']
        test_df = []
        for line in test_data:
            test_df.append({"concepts":line['meaning_representation'],"target":line['human_reference']})
    return test_df

def clean_answer(task, input_text):
    if task == 'nq' or task == 'triviaqa':
        clean_text = input_text.strip().split('\n')[0].split('<eoa>')[0].strip()
    elif task == 'e2e':
        clean_text = input_text.strip().split('\n')[0].split('<eoa>')[0].strip()
    elif task == 'addsub' or task == 'asdiv' or task == 'gsm8k':
        INVALID_ANS = "[invalid]"
        ANSWER_TRIGGER = "The answer is"
        input_text = input_text.lower()
        preds = input_text.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
        if len(pred) == 0:
            return INVALID_ANS
        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]
        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]
        clean_text = pred
    return clean_text
