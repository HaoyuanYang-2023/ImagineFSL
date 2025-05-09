export PYTHONPATH=. 

CLASSIFER='YOUR_VISUAL_CLASSIFIER_PATH'
ADAPTER='YOUR_ADAPTER_PATH'
TEXT='YOUR_TEXT_ENCODER_PATH'
PRETRAINED_WEIGHTS='YOUR_PRETRAIN_WEIGHTS_PATH'
FUSE_WEIGTH=THE_FUSE_WEIGHT
TEMPERATURE=THE_TEMPERATURE
DATASET='THE_DATASET'

python dinov2/eval/ct_tuning_mixing.py --clip-path clip/ViT-B-16.pt --num-workers 16 --config-file dinov2/configs/eval/clip_b16.yaml --pretrained-weights $PRETRAINED_WEIGHTS --output-dir ct_eval --category $DATASET --is_test True --eval_only True --classifier_path $CLASSIFER --adapter_path $ADAPTER --text_path $TEXT --fuse_weight $FUSE_WEIGTH --temperature $TEMPERATURE

