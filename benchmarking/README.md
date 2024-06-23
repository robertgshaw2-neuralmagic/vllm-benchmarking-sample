# Benchmark

This this example, we will demonstrate how to benchmark vLLM in the following modes:
- Inference Serving
- Offline Batch
- Specific Shapes

## Serving

When evaluating LLM performance in server mode, there are two latency metrics to consider:
- TTFT (Time to first token) measures how long it takes to generate the first token. 
- TPOT (Time per output token) measures how long it takes to generate each incremental token.

We will measure these by:
- Spinning up a vLLM server
- Spinning up clients and measure the metrics

### Spin Up vLLM Server

Install:

```bash
python -m venv vllm-venv
source vllm-venv/bin/activate
pip install vllm
```

Launch:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --disable-log-requests
```

### Spin Up Clients

Install:

```bash
python3 -m venv benchmark-venv
source benchmark-venv/bin/activate
pip install -U aiohttp transformers
```

Download sample data:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Launch the clients (we launch 1 client per second here):

```bash
python3 benchmark_serving.py \
    --backend openai \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --request-rate 1.0 \
    --num-prompts 200 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

Results:

We achieve `43ms` of TPOT on an A10.

```bash
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  221.06    
Total input tokens:                      45551     
Total generated tokens:                  39322     
Request throughput (req/s):              0.90      
Input token throughput (tok/s):          206.05    
Output token throughput (tok/s):         177.88    
---------------Time to First Token----------------
Mean TTFT (ms):                          103.74    
Median TTFT (ms):                        76.73     
P99 TTFT (ms):                           298.90    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          43.15     
Median TPOT (ms):                        42.29     
P99 TPOT (ms):                           66.84     
==================================================
```

## Offline Batch

When evaluating LLM performance in offline batch mode, we are focused on maximizing throughput.

Install:

```bash
python -m venv vllm-venv
source vllm-venv/bin/activate
pip install vllm
```

Run sample workload:

```bash
python benchmark_offline.py --help
python benchmark_offline.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

Results:

```bash
* ==========================================================
* Total Time:                   54.27
* Total Generations:            1000


* Generations / Sec:            18.43
* Generation Tok / Sec:         4198.43
* Prompt Tok / Sec:             10425.07


* Avg Generation Tokens:        227.85
* Avg Prompt Tokens:            565.78
* ==========================================================
```

We are able to procss 18.43 generations per second on an `H100 80GB HBM3`.

In this dataset, the average generations included:
- 566 prompt tokens
- 228 generation tokens.

## Specific Shapes

- example coming soon!