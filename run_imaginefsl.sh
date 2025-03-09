export PYTHONPATH=.

EXP_NAME=exp
PRETRAIIN_WEIGHTS=./output/clip_b16/checkpoint.pth
python dinov2/eval/imgainefsl_tuning_pipline.py --pretrain_weights $PRETRAIIN_WEIGHTS --exp_name $EXP_NAME

