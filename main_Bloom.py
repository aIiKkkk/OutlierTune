import torch
from Smoothquant.smooth import smooth_lm
from smooth_bloom import smooth_bloomlm
from fake_quant_bloom import W8A8Linear_QKVFc1, W8A8Linear_OFc2, W8A8Linear
from DecoderLayer import QuantOPTDecoderLayer
from tqdm import tqdm
import torch.nn as nn
from pprint import pprint
from bloom import BLOOMClass
from symmetrization import symmetrization_lm
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from datautils import get_loaders
import os


def quantize_model(model, act_scales, weight_quant='per_channel', act_quant='per_channel', quantize_bmm_input=True):
    # print(model)
    for i in tqdm(range(len(model.transformer.h))):
        qkv_input_scales = act_scales[f"transformer.h.{i}.self_attention.query_key_value"]["input"]
        q_output_scales = act_scales[f"transformer.h.{i}.self_attention.query_key_value"]["output"]
        o_input_scales = act_scales[f"transformer.h.{i}.self_attention.dense"]["input"]
        fc1_input_scales = act_scales[f"transformer.h.{i}.mlp.dense_h_to_4h"]["input"]
        fc2_input_scales = act_scales[f"transformer.h.{i}.mlp.dense_4h_to_h"]["input"]

        model.transformer.h[i].self_attention.query_key_value = W8A8Linear_QKVFc1.from_float(
            model.transformer.h[i].self_attention.query_key_value,
            input_scales=qkv_input_scales, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.transformer.h[i].self_attention.dense = W8A8Linear_OFc2.from_float(
            model.transformer.h[i].self_attention.dense,
            input_scales=o_input_scales, weight_quant=weight_quant, act_quant='per_token')

        model.transformer.h[i].mlp.dense_h_to_4h = W8A8Linear_QKVFc1.from_float(
            model.transformer.h[i].mlp.dense_h_to_4h,
            input_scales=fc1_input_scales, weight_quant=weight_quant, act_quant=act_quant)
        model.transformer.h[i].mlp.dense_4h_to_h = W8A8Linear_OFc2.from_float(
            model.transformer.h[i].mlp.dense_4h_to_h,
            input_scales=fc2_input_scales, weight_quant=weight_quant, act_quant='per_token')
    return model

def quantize_model_smooth(model, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True):
    for i in tqdm(range(len(model.transformer.h))):
        model.transformer.h[i].self_attention.query_key_value = W8A8Linear.from_float(
            model.transformer.h[i].self_attention.query_key_value, weight_quant=weight_quant, act_quant=act_quant,
            quantize_output=quantize_bmm_input)
        model.transformer.h[i].self_attention.dense = W8A8Linear.from_float(
            model.transformer.h[i].self_attention.dense, weight_quant=weight_quant, act_quant=act_quant)

        model.transformer.h[i].mlp.dense_h_to_4h = W8A8Linear.from_float(
            model.transformer.h[i].mlp.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant)
        model.transformer.h[i].mlp.dense_4h_to_h = W8A8Linear.from_float(
            model.transformer.h[i].mlp.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant)
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
            cache_testloader = f"{dataset}_testloader_{model_name}_all.cache"

            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                print(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(dataset, model=model_name,
                                                     cache_dir=f"./Model_data/{model_name}")
                torch.save(testloader, cache_testloader)
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
                outputs = model.transformer(batch)
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

model_name = "bloom-7b1"
tokenizer = AutoTokenizer.from_pretrained(f"./Model_data/{model_name}")
dataset = load_dataset('lambada_openai', split='validation[:1000]')
kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
evaluator_PPL = Evaluator(dataset, tokenizer, 'cuda')

compare_oursw8a8 = True
if compare_oursw8a8:
    model = AutoModelForCausalLM.from_pretrained(f"./Model_data/{model_name}", **kwargs)
    scales = torch.load(f'./symmetrizations/{model_name}.pt')
    hyperparameters = False
    symmetrization_lm(model, scales, hyperparameters)
    act_scales = torch.load(f'./act_scales_sym/{model_name}.pt')
    smooth_lm(model, act_scales)
    print("Starting quantize_activations")
    model_smoothquant_w8a8 = quantize_model(model, act_scales)


    acc_smoothquant_w8a8 = evaluator_PPL.evaluate(model_smoothquant_w8a8, model_name)
