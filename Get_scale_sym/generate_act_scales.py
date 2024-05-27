import torch
import os
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
from smooth_bloom import smooth_bloomlm
from act_calibration import get_static_decoder_layer_scales
from symmetrization import symmetrization_lm

hyperparameters = 0.000000001
def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}", model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(f"/data/wangjinguang/Model_data/{model_name}", **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    model_name = 'Llama-2-7b'
    parser.add_argument('--model-name', type=str,
                        default=model_name, help='model name')
    parser.add_argument('--output-path', type=str,
                        default=f'/home/wjg/linuxPJ/New/act_scales_sym/{model_name}.pt',
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
    if "Llama" in args.model_name:
        # symmetrization_lm(model, scales = None, hyperparameters = None)
        act_scales = torch.load(f'/home/wjg/linuxPJ/New/act_scales_or/{args.model_name}.pt')
        smooth_bloomlm(model, act_scales, 0.35)
    else:
        scales = torch.load(f'/home/wjg/linuxPJ/New/symmetrizations/{args.model_name}.pt')

        symmetrization_lm(model, scales, hyperparameters)
    act_scales = get_static_decoder_layer_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)



if __name__ == '__main__':
    main()