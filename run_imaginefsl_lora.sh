export PYTHONPATH=.

EXP_NAME='exp'
PRETRAIIN_WEIGHTS=YOUR_PRETRAIN_WEIGHTS_PATH
python dinov2/eval/imaginefsl_lora_tuning_pipline.py --pretrain-weights $PRETRAIIN_WEIGHTS --exp_name $EXP_NAME



