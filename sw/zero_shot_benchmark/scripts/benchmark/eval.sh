shots=0
name_for_output_file=qwem2.5-3b-0-shot
model=Qwen/Qwen2.5-3B
model_arch=qwen
for task in boolq hellaswag
do
python -u ../../evaluate_task_result.py \
    --result-file results/${task}-${shots}-${name_for_output_file}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots} \
    --model-type ${model_arch}
done
