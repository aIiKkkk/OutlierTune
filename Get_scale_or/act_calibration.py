import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


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
            plot = False
            if plot:
                Y = torch.arange(4096)
                X = torch.arange(333)
                X1, Y1 = torch.meshgrid(X, Y)
                Z = x[0].to("cpu").numpy()
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X1, Y1, Z, cmap='viridis')
                ax.set_xlabel('Column Index')
                ax.set_ylabel('Row Index')
                ax.set_zlabel('Value')
                ax.set_title('3D Surface Plot of Tensor')
                fig.colorbar(surf)
                plt.show()

            min_value, max_value = torch.min(x), torch.max(x)
            min_val_cur = min_value * 0.8
            max_val_cur = max_value * 0.8
            clipped_x = torch.clamp(x, min=min_val_cur, max=max_val_cur)
            act_dict[name]["input"] = clipped_x.detach().abs().amax(-2, keepdim=True)
        else:
            min_value, max_value = torch.min(x), torch.max(x)
            min_val_cur = min_value * 0.8
            max_val_cur = max_value * 0.8
            clipped_x = torch.clamp(x, min=min_val_cur, max=max_val_cur)
            new_max_values = clipped_x.detach().abs().amax(-2, keepdim=True)
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

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset('json', data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    #cache_dataloader = (f"/home/wjg/linuxPJ/smoothquant-main/Scale_get/dataloader_opt_mix_128_mix.cache")
    #dataloader = torch.load(cache_dataloader)
    #for batch in tqdm(dataloader):

    #    model(batch[0].to(device))
    for i in pbar:
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
    #    print(input_ids.shape)
        model(input_ids)
    for hook in hooks:
        hook.remove()

    return act_dict
