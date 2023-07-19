export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1 
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp_v100.yaml \
python pipeline/train/continual.py \
--pretrained_model_name_or_path=luodian/OTTER-MPT7B-Init  \
--mimicit_path="/mnt/lustre/dwzhou/data/gpt/m3it/vqa/st-vqa/st-vqa_instructions.json" \
--images_path="/mnt/lustre/dwzhou/data/gpt/m3it/vqa/st-vqa/st-vqa_00.json" \
--train_config_ic_path="/mnt/lustre/dwzhou/data/gpt/m3it/vqa/st-vqa/st-vqa_train.json" \
--batch_size=4 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=ntu-slab \
--run_name=OTTER-LLaMA7B-densecaption \
--wandb_project=OTTER-LLaMA7B-3mpt  \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--model_name='otter_finetune' \
--dataset='vqa' \
--data_seed=1993 \
--convnet_type='flamingo' \
