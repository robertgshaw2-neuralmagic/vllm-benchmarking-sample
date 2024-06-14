# vLLM Acceleration with Marlin

This this example, we will demonstrate how to benchmark vLLM. 

We will compare the speed of an Fp16 model to an Int4 model using Marlin kernels for inference acceleration.

Check out our [technical blog](https://neuralmagic.com/blog/pushing-the-boundaries-of-mixed-precision-llm-inference-with-marlin/) for more details on how Marlin works.

> Note: this example requires Ampere GPUs or later.

## Launch 4 Bit Model

We can use `nm-testing/Meta-Llama-3-8B-Instruct-GPTQ`, which is posted on the Hugging Face hub.

Download the docker image to get started:

### Install

Install vLLM:

```bash
python -m venv env
source env/bin/activate
pip install vllm
```

### Deploy INT4 Model

Launch:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model nm-testing/Meta-Llama-3-8B-Instruct-GPTQ \
    --disable-log-requests
```

### Benchmark the INT4 Model

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

We can see that the quantized models gets `7.79ms` TPOT.

```bash
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  201.45    
Total input tokens:                      45551     
Total generated tokens:                  39243     
Request throughput (req/s):              0.99      
Input token throughput (tok/s):          226.11    
Output token throughput (tok/s):         194.80    
---------------Time to First Token----------------
Mean TTFT (ms):                          28.04     
Median TTFT (ms):                        19.79     
P99 TTFT (ms):                           87.08     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.79      
Median TPOT (ms):                        7.66      
P99 TPOT (ms):                           11.86     
==================================================
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

We can see the fp16 models get `39ms` TPOT, about 3.5x slower than the INT4 model.

```bash
---------------Time to First Token----------------
Mean TTFT (ms):                          97.13     
Median TTFT (ms):                        66.12     
P99 TTFT (ms):                           311.50    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          38.85     
Median TPOT (ms):                        39.36     
P99 TPOT (ms):                           47.97     
==================================================
```
