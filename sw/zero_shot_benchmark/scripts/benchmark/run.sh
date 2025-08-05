shots=0
name_for_output_file=qwem2.5-3b-0-shot
model=Qwen/Qwen2.5-3B # model path
local_saved_model=/workspace/release_repos/FlexCiM/sw/models/qwen_prune_0.6.pth # If you want to use a local pruned model, please specify the path here. If not have it the same as model name
python -u ../../run_benchmarking.py \
    --output-path results/task-${shots}-${name_for_output_file}.jsonl \
    --model-name ${model} --local-saved-model ${local_saved_model}
