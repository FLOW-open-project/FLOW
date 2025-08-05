python finetune.py \
    --base_model <HF-model name> \
    --local_saved_model /workspace/.../FlexCiM/sw/models/*.pth \
    --data_path './alpaca_data_gpt4.json' \
    --output_dir './llama_results' \
    --batch_size 128 \
    --cutoff_len 512\
    --micro_batch_size 8 \
    --num_epochs 3