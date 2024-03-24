# WANDB_MODE=offline \
    # --model_name_or_path roberta-base \
# CUDA_VISIBLE_DEVICES=1,2,3 \
    # --model_name_or_path bert-base-uncased \
# CUDA_VISIBLE_DEVICES=1,2,3 \
# CUDA_VISIBLE_DEVICES=1 \
#  --dataset_name wikitext \
# CUDA_VISIBLE_DEVICES=1,2,3 \

WANDB_PROJECT=recipetrend_test1 \
WANDB_MODE=online \
CUDA_VISIBLE_DEVICES=0 \
python run_recipebuild.py \
    --run_name recipebuild_set_10K \
    --model_type bert \
    --tokenizer_name bert-base-uncased \
    --max_seq_length 25 \
    --train_file /home/jaesung/jaesung/research/recipetrend/preprocess/recipe_set_10K_train.txt \
    --validation_file /home/jaesung/jaesung/research/recipetrend/preprocess/recipe_set_10K_val.txt \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --do_train \
    --do_eval \
    --output_dir ./checkpoints/recipebuild \
    --overwrite_output_dir \
    --logging_steps 100 \
    --log_level info \
    --evaluation_strategy steps \
    --num_train_epochs 30 \
    --save_steps 500 \
    --line_by_line true \
    --pad_to_max_length true \
    --config_overrides num_attention_heads=1,num_hidden_layers=1 \
    
    # --max_steps=200 \
    # --learning_rate


