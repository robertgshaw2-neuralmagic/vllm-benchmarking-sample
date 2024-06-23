import argparse
import torch
from datasets import load_dataset
from sparseml.transformers import SparseAutoModelForCausalLM, oneshot
from transformers import AutoTokenizer

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True)
parser.add_argument("--save-dir", type=str, required=True)
parser.add_argument("--num-calibration-samples", type=int, default=512)
parser.add_argument("--max-sequence-length", type=int, default=2048)

if __name__ == "__main__":
    # Parse arguments.
    args = parser.parse_args()
    model_id = args.model_id
    save_dir = args.save_dir
    num_calibration_samples = args.num_calibration_samples
    max_sequence_length = args.max_sequence_length

    # Load model.
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Format dataset.
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(num_calibration_samples))
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False,
        )}
    ds = ds.map(preprocess)

    # BE CAREFUL WITH THE TOKENIZER
    #   apply_chat_template already adds the bos_token for meta-llama
    #   so we set add_special_token to false
    examples = [
        tokenizer(
            example["text"],
            padding=False, max_length=max_sequence_length, truncation=True, add_special_tokens=False
        ) for example in ds
    ]

    # Configure algorithms.
    recipe = """
    quant_stage:
        quant_modifiers:
            SmoothQuantModifier:
                smoothing_strength: 0.8
                mappings: [
                    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
                ]
            GPTQModifier:
                sequential_update: false
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 8
                            type: "int"
                            symmetric: true
                            strategy: "channel"
                        input_activations:
                            num_bits: 8
                            type: "int"
                            symmetric: true
                            dynamic: true
                            strategy: "token"
                        targets: ["Linear"]
    """

    # Apply algorithms.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
    )

    # Confirm generations of the quantized model look sane.
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))

    # Save.
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)