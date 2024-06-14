# vLLM Acceleration with Quantization (Marlin)

This this example, we will demonstrate how to benchmark vLLM. 

We will compare the speed of an Fp16 model to an Int4 model using Marlin kernels for inference acceleration. Check out our [technical blog](https://neuralmagic.com/blog/pushing-the-boundaries-of-mixed-precision-llm-inference-with-marlin/) for more details on how Marlin works.

> Note: this example requires Ampere GPUs or later.

## Install

Install vLLM:

```bash
python -m venv env
source env/bin/activate
pip install vllm
```

## Benchmark 4bit Model

We can use `nm-testing/Meta-Llama-3-8B-Instruct-GPTQ`, which is posted on the Hugging Face hub.

### Deploy

Launch:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model nm-testing/Meta-Llama-3-8B-Instruct-GPTQ \
    --disable-log-requests
```

### Benchmark

When evaluating LLM performance, there are two latency metrics to consider. 
- TTFT (Time to first token) measures how long it takes to generate the first token. 
- TPOT (Time per output token) measures how long it takes to generate each incremental token.

The benchmark scripts provided here help us to evaluate these metrics.

#### Install
Install dependencies to run the benchmark client.

```bash
python3 -m venv benchmark-env
source benchmark-env/bin/activate
pip install -U aiohttp transformers
```

#### Run the Benchmark

Download some sample data:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Launch the benchmark script. We run 1 query per second.

```bash
python3 benchmark_serving.py \
    --backend openai \
    --model nm-testing/Meta-Llama-3-8B-Instruct-GPTQ \
    --request-rate 1.0 \
    --num-prompts 200 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

### Deploy FP16 Model

Now, let's deploy the FP16 model.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --disable-log-requests
```

### Benchmark the FP16 Model

We can use the same script, just swapping out the model. We run 1 query per second.

```bash
python3 benchmark_serving.py \
    --backend openai \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --request-rate 1.0 \
    --num-prompts 200 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```
