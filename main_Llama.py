import torch
from Smoothquant.smooth import smooth_lm
from smooth_bloom import smooth_bloomlm
from fake_quant_Llama import W8A8Linear_QKVFc1, W8A8Linear, W8A8Linear_OFc2_Llama
from DecoderLayer import QuantOPTDecoderLayer
from tqdm import tqdm
import torch.nn as nn
from pprint import pprint
from bloom import BLOOMClass
from datautils import get_loaders
from symmetrization import symmetrization_lm
from datasets import load_from_disk , load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM)
import gc
import os


def quantize_model(model, act_scales, weight_quant='per_channel', act_quant='per_channel', quantize_bmm_input=True):
    for i in tqdm(range(len(model.model.layers))):
        qkv_input_scales = act_scales[f"model.layers.{i}.self_attn.q_proj"]["input"]
        fc1_input_scales = act_scales[f"model.layers.{i}.mlp.gate_proj"]["input"]

        model.model.layers[i].self_attn.q_proj = W8A8Linear_QKVFc1.from_float(
            model.model.layers[i].self_attn.q_proj,
            input_scales=qkv_input_scales, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.k_proj = W8A8Linear_QKVFc1.from_float(
            model.model.layers[i].self_attn.k_proj,
            input_scales=qkv_input_scales, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.v_proj = W8A8Linear_QKVFc1.from_float(
            model.model.layers[i].self_attn.v_proj,
            input_scales=qkv_input_scales, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.o_proj = W8A8Linear_OFc2_Llama.from_float(
            model.model.layers[i].self_attn.o_proj, weight_quant=weight_quant, act_quant='per_token')


        model.model.layers[i].mlp.gate_proj = W8A8Linear_QKVFc1.from_float(model.model.layers[i].mlp.gate_proj,
            input_scales=fc1_input_scales, weight_quant=weight_quant, act_quant=act_quant)
        model.model.layers[i].mlp.up_proj = W8A8Linear_QKVFc1.from_float(model.model.layers[i].mlp.up_proj,
            input_scales=fc1_input_scales, weight_quant=weight_quant, act_quant=act_quant)
        model.model.layers[i].mlp.down_proj = W8A8Linear_OFc2_Llama.from_float(
            model.model.layers[i].mlp.down_proj, weight_quant=weight_quant, act_quant='per_token')
    return model

def quantize_model_smooth(model, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True):
    for i in tqdm(range(len(model.model.layers))):
        model.model.layers[i].self_attn.q_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.q_proj, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.k_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.k_proj, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.v_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.v_proj, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.model.layers[i].self_attn.o_proj = W8A8Linear.from_float(
            model.model.layers[i].self_attn.o_proj, weight_quant=weight_quant, act_quant=act_quant)


        model.model.layers[i].mlp.gate_proj = W8A8Linear.from_float(
            model.model.layers[i].mlp.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
        model.model.layers[i].mlp.up_proj = W8A8Linear.from_float(
            model.model.layers[i].mlp.up_proj, weight_quant=weight_quant, act_quant=act_quant)
        model.model.layers[i].mlp.down_proj = W8A8Linear.from_float(
            model.model.layers[i].mlp.down_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model



class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model, model_name):
        model.eval()

        for dataset in ["wikitext2", "ptb", "c4"]:
            if "Llama-2" in model_name:
                cache_testloader = f"{dataset}_testloader_Llama-2_all.cache"
            elif "Llama-3" in model_name:
                cache_testloader = f"{dataset}_testloader_Llama-3_all.cache"
            if os.path.exists(cache_testloader):

                testloader = torch.load(cache_testloader)
                print(f"load calibration from {cache_testloader}")
            else:
                # testloader = load_dataset('json', data_files="/data/wangjinguang/dataset/c4-05-10/c4-validation.00000-of-00008.json.gz", split='train')
                # testloader = testloader.shuffle(seed=42)
                trainloader, testloader = get_loaders(dataset, model=model_name,
                                                      cache_dir=f"./Model_data/{model_name}")
                torch.save(testloader, cache_testloader)
                # print(testloader)
            if "c4" == dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids
            seqlen = 2048
            nsamples = testenc.numel() // seqlen
            model.config.use_cache = False
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(model.device)
                outputs = model.model(batch)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                               :, 1:
                               ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            print(dataset, ppl.item())
        return ppl

model_name = "Llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(f"./Model_data/{model_name}")
dataset = load_dataset('lambada_openai', split='validation[:1000]')

evaluator_PPL = Evaluator(dataset, tokenizer, 'cuda')

compare_oursw8a8 = True
if compare_oursw8a8:
    model = AutoModelForCausalLM.from_pretrained(f"./Model_data/{model_name}",
                                                      torch_dtype=torch.float16, device_map='auto')
    act_scales = torch.load(f'./act_scales_or/{model_name}-0.8.pt')
    smooth_lm(model, act_scales)
    model_ours_w8a8 = quantize_model(model, act_scales)
    print("Starting evaluate")
    acc_smoothquant_w8a8 = evaluator_PPL.evaluate(model_ours_w8a8, model_name)
