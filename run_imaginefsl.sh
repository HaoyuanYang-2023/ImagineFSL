export PYTHONPATH=.

EXP_NAME='exp'
PRETRAIIN_WEIGHTS=YOUR_PRETRAIN_WEIGHTS_PATH
python dinov2/eval/imgainefsl_tuning_pipline.py --pretrain_weights $PRETRAIIN_WEIGHTS --exp_name $EXP_NAME

