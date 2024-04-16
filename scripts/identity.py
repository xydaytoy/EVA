from transformers import AutoTokenizer
import sys
root="/data/xyyf/EVA"
sys.path.append(root)
import model_info
from model_info import models, vocab_sizes
import argparse

save_file = "/data/xyyf/EVA/map_file/"
def main(args):
    model_name1 = args.model1#"llama2"
    model_ckpt1 = models[model_name1]
    tokenizer1 = AutoTokenizer.from_pretrained(model_ckpt1, trust_remote_code=True)
    src = set(tokenizer1.get_vocab().keys())
    src_dict = tokenizer1.get_vocab()

    model_name2 = args.model2#"baichuan2"
    model_ckpt2 = models[model_name2]
    tokenizer2 = AutoTokenizer.from_pretrained(model_ckpt2, trust_remote_code=True)
    tgt = set(tokenizer2.get_vocab().keys())
    tgt_dict = tokenizer2.get_vocab()

    common_elements = src.intersection(tgt)

    with open(save_file+model_name1+"-"+model_name2+".dict","w") as f:
        for i in common_elements:
            f.write(str(src_dict[i]) + " " + str(tgt_dict[i]) + "\n")
        if model_name1 == "chatglm2" or model_name2 == "chatglm2":
            f.write(str(0) + " " + str(0) + "\n")
            f.write(str(1) + " " + str(1) + "\n")
            f.write(str(2) + " " + str(2) + "\n")
    with open(save_file+model_name1+"-"+model_name2+"-test.dict","w") as f:
        for i in range(vocab_sizes[model_name1]):
            f.write(str(i) + " " + str(0) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--model1", type=str)
parser.add_argument("--model2", type=str)

args = parser.parse_args()

main(args)