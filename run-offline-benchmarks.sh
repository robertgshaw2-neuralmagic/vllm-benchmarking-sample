MODELS=("meta-llama/Meta-Llama-3-8B-Instruct" "nm-testing/Meta-Llama-3-8B-Instruct-GPTQ" "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dyn-Per-Token")
MAX_NUM_SEQS=256

for model in ${MODELS[@]}; do
    echo "\n\n"
    echo "* ==================================================================================== *"

    echo "* === RUNNING ${MODEL} WITH MAX_NUM_SEQS=${MAX_NUM_SEQS}"
    python benchmark/benchmark_offline.py --model ${MODEL} --max-num-seqs ${MAX_NUM_SEQS}
    
    echo "* ==================================================================================== *"
    echo "\n\n"
done