# OutlierTune: Efficient Channel-Wise Quantization for Large Language Models

## Abstract

Quantizing the activations  of large language models (LLMs) has been a significant challenge due to the presence of structured outliers. Most existing methods focus on the per-token or per-tensor quantization of activations, making it difficult to achieve both accuracy and hardware efficiency. To address this problem, we propose OutlierTune, an efficient per-channel post-training quantization (PTQ) method for the activations of LLMs. OutlierTune consists of two  components: pre-execution of dequantization and symmetrization. The pre-execution of dequantization updates the model weights by the activation scaling factors, avoiding the internal scaling and costly additional computational overheads brought by the per-channel activation quantization. The symmetrization further reduces the quantization differences arising from the weight updates by ensuring the balanced numerical ranges across different activation channels. OutlierTune is easy to implement and hardware-efficient, introducing almost no additional computational overheads during the inference. Extensive experiments show that the proposed framework outperforms existing methods across multiple different tasks. Demonstrating better generalization, this framework improves the Int6 quantization of the instruction-tuning LLMs, such as OPT-IML, to the same level as half-precision (FP16). Moreover, we have  shown that the proposed framework is 1.48$\times$ faster than the FP16 implementation while reducing approximately 2$\times$ memory usage.

## Installation

```bash
conda create -n smoothquant python=3.8
conda activate smoothquant
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers accelerate datasets zstandard

python setup.py install
```

### Activation Scales and Calibration

We provide the activation channel scales for OPT and BLOOM models in [act_scales/](act_scales/). We get those scales with 512 random sentences in the Pile validation set.

We also provide the script to get the activation channel scales for your models. Please refer to [examples/generate_act_scales.py](examples/generate_act_scales.py). You can use the following command to get the scales for your models:
