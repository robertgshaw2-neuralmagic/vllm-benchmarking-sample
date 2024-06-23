import argparse
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
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

    # Setup quantization arguments.
    #   We also support speedup from 8 bits.
    #   With 8 bits, its best to set group_size=-1 (channelwise) / desc_act=False)
    quantize_config = BaseQuantizeConfig(
        bits=4,                         # Weights quantized to 4 bit.
        group_size=128,                 # Group size 128 is typically the best spot for accuracy / performance.
        desc_act=True,                  # Act_recordering will help accuracy.
        model_file_base_name="model",   # Name of the model.safetensors when we call save_pretrained
        true_sequential=False,          # Set to true to reduce memory consumption for slower runtime of .quantize()
    )

    # Load model.
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config,
        device_map="auto")
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

    # Apply the GPTQ algorithm.
    model.quantize(examples)

    # Save Model To Disk.
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)