import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,AutoModelForSeq2SeqLM)
import argparse

from sym_calibration import get_static_decoder_layer_symmetrizations

device_map={'glm.word_embeddings': 0,
 'glm.transformer.embedding_dropout': 0,
 'glm.transformer.position_embeddings': 0,
 'glm.transformer.block_position_embeddings': 0,
 'glm.transformer.layers.0': 0,
 'glm.transformer.layers.1': 0,
 'glm.transformer.layers.2': 0,
 'glm.transformer.layers.3': 0,
 'glm.transformer.layers.4': 0,
 'glm.transformer.layers.5': 0,
 'glm.transformer.layers.6': 0,
 'glm.transformer.layers.7': 0,
 'glm.transformer.layers.8': 0,
 'glm.transformer.layers.9': 0,
 'glm.transformer.layers.10': 0,
 'glm.transformer.layers.11': 0,
 'glm.transformer.layers.12': 0,
 'glm.transformer.layers.13': 0,
 'glm.transformer.layers.14': 0,
 'glm.transformer.layers.15': 0,
 'glm.transformer.layers.16': 0,
 'glm.transformer.layers.17': 0,
 'glm.transformer.layers.18': 0,
 'glm.transformer.layers.19': 0,
 'glm.transformer.layers.20': 0,
 'glm.transformer.layers.21': 0,
 'glm.transformer.layers.22': 0,
 'glm.transformer.layers.23': 0,
 'glm.transformer.layers.24': 0,
 'glm.transformer.layers.25': 0,
 'glm.transformer.layers.26': 0,
 'glm.transformer.layers.27': 0,
 'glm.transformer.layers.28': 0,
 'glm.transformer.layers.29': 0,
 'glm.transformer.layers.30': 0,
 'glm.transformer.layers.31': 0,
 'glm.transformer.layers.32': 0,
 'glm.transformer.layers.33': 0,
 'glm.transformer.layers.34': 0,
 'glm.transformer.layers.35': 0,
 'glm.transformer.layers.36': 0,
 'glm.transformer.layers.37': 0,
 'glm.transformer.layers.38': 0,
 'glm.transformer.layers.39': 0,
 'glm.transformer.layers.40': 0,
 'glm.transformer.layers.41': 0,
 'glm.transformer.layers.42': 0,
 'glm.transformer.layers.43': 0,
 'glm.transformer.layers.44': 0,
 'glm.transformer.layers.45': 0,
 'glm.transformer.layers.46': 0,
 'glm.transformer.layers.47': 0,
 'glm.transformer.final_layernorm': 0}
def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}", trust_remote_code=True)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForSeq2SeqLM.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}",
                                                  torch_dtype=torch.float16, trust_remote_code=True,
                                                  device_map=device_map, revision="6adb492")
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    model_name = 'GLM-10b'
    parser.add_argument('--model-name', type=str,
                        default=model_name, help='model name')
    parser.add_argument('--output-path', type=str, default=f'/home/wjg/linuxPJ/New/symmetrizations/{model_name}.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='/home/wjg/linuxPJ/smoothquant-main/dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    act_scales = get_static_decoder_layer_symmetrizations(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)
    # print(act_scales[f"model.decoder.layers.{0}.self_attn.q_proj"]['input'].shape)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)

    # import matplotlib.pyplot as plt
    # tensor_to_visualize = act_scales[f"model.decoder.layers.{0}.self_attn.q_proj"]['input']
    # tensor_data = tensor_to_visualize.cpu().detach().numpy().flatten()
    # plt.hist(tensor_data, bins=50, alpha=0.7, color='blue')
    # plt.title('Histogram of attn_input_scale')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()


if __name__ == '__main__':
    main()
