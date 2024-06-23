# Apply Quantization

In this example, we will look at how to apply quantization to Llama-3-8B.

## Weight Only Quantization

We can use `auto-gptq` to quantize the weights of the model.

Install:

```bash
python -m venv autogptq-venv
source autogptq-venv/bin/activate
pip install -r requirements-autogptq.txt
```

Apply Quantization:

```bash
python3 w4a16-example.py --help
python3 w4a16-example.py --model-id meta-llama/Meta-Llama-3-8B-Instruct --save-dir Meta-Llama-3-8B-Instruct-W4A16
```

## Weight and Activation Quantization

We can use `sparseml` to quantize the weights and activations of the model.

Install:

```bash
python -m venv sparseml-venv
source sparseml-venv/bin/activate
pip install -r requirements-sparseml.txt
```

Apply Quantization:

```bash
python3 w8a8-example.py --help
python3 w8a8-example.py --model-id meta-llama/Meta-Llama-3-8B-Instruct --save-dir Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
```