export PYTHONPATH=.

EXP_NAME=exp
PRETRAIIN_WEIGHTS=PATH_TO_PRETRAINED_WEIGHTS.pth
python dinov2/eval/imgainefsl_tuning_pipline.py --pretrain_weights $PRETRAIIN_WEIGHTS --exp_name $EXP_NAME

