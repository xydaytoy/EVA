# EVA (Bridging the Gap between Different Vocabularies for LLM Ensemble - NAACL 2024)
Source code for the NAACL 2024 long paper [Bridging the Gap between Different Vocabularies for LLM Ensemble](https://openreview.net/forum?id=IOG33p5r6o).

## Contents
* [Usage](#usage)
* [Requirements](#requirements)
* [Citation](#citation)

## Usage

### Information file

+ model_info.py
+ task_info.py

### Step1: Cross-Model Vocabulary Alignment

+ Get the files required for Vecmap input

  ```shell
  python identity.py --model1 llama2 --model2 baichuan2
  python hidden.py --model llama2
  python hidden.py --model baichuan2
  sed -i '1s/^/32000 4096\n/' llama2.emb
  sed -i '1s/^/125696 4096\n/' baichuan2.emb
  ```

+ Vocabulary projection 

  ```shell
  model1=llama2
  model2=baichuan2
  python /data/xyyf/EVA/vecmap/map_embeddings.py --supervised $root/map_file/${model1}-${model2}.dict $root/map_idx_all/${model1}.emb $root/map_file/${model2}.emb $root/map_file/${model1}_${model2}/${model1}_mapped_sup.emb $root/map_file/${model1}_${model2}/${model2}_mapped_sup.emb
  ```

+ Get similarity matrix

  ```shell
  model1=llama2
  model2=baichuan2
  python /data/xyyf/EVA/vecmap/eval_translation_scipy_matrix.py $root/map_file/${model1}_${model2}/${model1}_mapped_sup.emb $root/map_file/${model1}_${model2}/${model2}_mapped_sup.emb -d $root/map_file/${model1}-${model2}-test.dict --retrieval csls --cuda --neighborhood 1 --precision fp32
  ```

### Step2: LLMs Ensemble

+ Suppose you want to ensemble Llama2-7b-chat-hf, Chatglm2-6b, Internlm-7b-chat and Baichuan2-7b-chat:
```shell
python eva_multi.py --task addsub --model baichuan2 --aux_models llama2 chatglm2 internlm --aux_method drop --matrix_name filter --topk 320 --drop 3
```

## Requirements

+ torch==2.0.1
+ transformers==4.28.1
+ spicy==3.6.0
+ [VecMap](https://github.com/artetxem/vecmap)

## Citation

Please cite the following paper if you use the code:

```
@inproceedings{
xu2024bridging,
title={Bridging the Gap between Different Vocabularies for {LLM} Ensemble},
author={Yangyifan Xu and Jinliang Lu and Jiajun Zhang},
booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=IOG33p5r6o}
}
```