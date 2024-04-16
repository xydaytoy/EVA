import sys
root="/data/xyyf/EVA"
sys.path.append(root)
import model_info
from model_info import models, module_names, vocab_sizes
from transformers import AutoModelForCausalLM
import argparse

def main(args):
    model_name=args.model
    model_ckpt = models[model_name]
    specific_module_name = module_names[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, trust_remote_code=True,low_cpu_mem_usage=True)
    specific_module_parameters = model.state_dict()[specific_module_name].numpy()

    save_file = "/data/xyyf/EVA/map_file/"
    with open(save_file+ model_name +".emb","w") as f:
        for i in range(vocab_sizes[model_name]):
            token = str(i)
            vector = specific_module_parameters[i]
            f.write(token + " " + " ".join(map(str, vector.flatten())) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)

args = parser.parse_args()

main(args)









