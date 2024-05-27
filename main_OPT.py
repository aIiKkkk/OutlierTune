import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

from Smoothquant.smooth import smooth_lm
from DecoderLayer import QuantOPTDecoderLayer
from tqdm import tqdm
import torch.nn as nn
import os
from pprint import pprint
from opt import OPTClass
from symmetrization import symmetrization_lm
from datasets import load_from_disk

def quantize_model(model, act_scales, weight_quant='per_token', act_quant='per_channel', quantize_bmm_input=True):
    embed_dim = model.model.decoder.layers[0].embed_dim
    num_heads = model.model.decoder.layers[0].self_attn.num_heads
    for i in tqdm(range(len(model.model.decoder.layers))):
        model.model.decoder.layers[i] = QuantOPTDecoderLayer(model.model.decoder.layers[i],
                             embed_dim, num_heads, weight_quant, act_quant, quantize_bmm_input, model.model.config, act_scales, i)
    torch.cuda.empty_cache()
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
    def evaluate(self, model):
        model.eval()

        for dataset in ["wikitext2", "ptb", "c4"]:

            cache_testloader = f"{dataset}_testloader_opt_all.cache"

            if os.path.exists(cache_testloader):

                testloader = torch.load(cache_testloader)
                print(f"load calibration from {cache_testloader}")
            else:
                trainloader, testloader = get_loaders(dataset, model=model_name,
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
                outputs = model.model.decoder(batch)
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

model_name = "opt-6.7b"

tokenizer = GPT2Tokenizer.from_pretrained(f"./Model_data/{model_name}")
dataset = load_dataset('lambada_openai', split='validation[:1000]')

evaluator_PPL = Evaluator(dataset, tokenizer, 'cuda')

model = OPTForCausalLM.from_pretrained(f"./Model_data/{model_name}",
                                        torch_dtype=torch.float16, device_map='auto')

print(model.model.decoder.layers[0].self_attn.k_proj.weight.dtype)
scales = torch.load(f'./symmetrizations/{model_name}.pt')

symmetrization_lm(model, scales)

act_scales = torch.load(f'./act_scales_sym/{model_name}.pt')
smooth_lm(model, act_scales)
print("Starting quantize_activations")
model_smoothquant_w8a8 = quantize_model(model, act_scales)

print("Starting evaluate")
acc_smoothquant_w8a8 = evaluator_PPL.evaluate(model_smoothquant_w8a8)


