export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

MODEL=YOUR_VISION_MODEL_PATH
CLASSIFER=YOUR_VISION_CLASSIFIER_PATH
ADAPTER=YOUR_ADAPTER_PATH
TEXT=YOUR_TEXT_CLASSIFIER_PATH
PRETRAINED_WEIGHTS=YOUR_PRETRAINED_WEIGHTS_PATH
FUSE_WEIGTH=THE_FUSE_WEIGHT_REFER_TO_README_TEXT
RANK=THE_RANK_REFER_TO_README_TEXT

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
--category dtd \
--lora_r $RANK --lora_start_block 0 --is_test True \
--fuse_weight $FUSE_WEIGTH \

