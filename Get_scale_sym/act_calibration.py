import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict
from Llama_LN_bias import LlamaLinear_bias
from functools import partial
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def get_static_decoder_layer_scales(model,
                                    tokenizer,
                                    dataset_path,
                                    num_samples=512,
                                    seq_len=512,
                                    ):
    model.eval()
    device = next(model.parameters()).device
    # print(model)
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):

        if isinstance(x, tuple):
            x = x[0]
        # print(x.shape)
        if name not in act_dict or "input" not in act_dict[name]:
            # 计算行维度的最大值并保持维度结构
            act_dict[name]["input"] = x.detach().abs().amax(-2, keepdim=True)
        else:
            # 计算行维度的最大值并保持维度结构
            new_max_values = x.detach().abs().amax(-2, keepdim=True)
            # 更新最大值
            act_dict[name]["input"] = torch.max(act_dict[name]["input"], new_max_values)

        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().amax(-2, keepdim=True)
        else:
            act_dict[name]["output"] = torch.max(
                act_dict[name]["output"], y.detach().abs().amax(-2, keepdim=True))

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))
        elif isinstance(m, LlamaLinear_bias):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset('json', data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    for hook in hooks:
        hook.remove()

    return act_dict
