export PYTHONPATH=.

MODEL='YOUR_VISUAL_MODEL_PATH'
CLASSIFER='YOUR_VISUAL_CLASSIFIER_PATH'
ADAPTER='YOUR_ADAPTER_PATH'
TEXT='YOUR_TEXT_ENCODER_PATH'
PRETRAINED_WEIGHTS='YOUR_PRETRAIN_WEIGHTS_PATH'
FUSE_WEIGTH=THE_FUSE_WEIGHT
RANK=THE_RANK
SCALE=THE_SCALE
DATASET='THE_DATASET'

python dinov2/eval/ct_lora_tuning_mixing.py \
--visual_classifier_resume $CLASSIFER \
--text_classifier_resume  $TEXT \
--visual_adapter_resume $ADAPTER \
--visual_model_resume $MODEL \
--eval_only True \
--config-file dinov2/configs/eval/clip_b16.yaml \
--pretrained-weights $PRETRAINED_WEIGHTS \
--clip-path clip/ViT-B-16.pt \
--output-dir ./output \
--is_synth \
--category $DATASET \
--lora_r $RANK --lora_scale $SCALE --lora_start_block 0 --is_test True \
--fuse_weight $FUSE_WEIGTH \

