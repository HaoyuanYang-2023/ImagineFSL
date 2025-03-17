export PYTHONPATH=. 
export CUDA_VISIBLE_DEVICES=0

TEMPERATURE=THE_TEMPERATURE_REFER_TO_README_TEXT
CLASSIFER=YOUR_VISION_CLASSIFIER_PATH
ADAPTER=YOUR_ADAPTER_PATH
TEXT=YOUR_TEXT_CLASSIFIER_PATH
PRETRAINED_WEIGHTS=YOUR_PRETRAINED_WEIGHTS_PATH
FUSE_WEIGTH=THE_FUSE_WEIGHT_REFER_TO_README_TEXT

python dinov2/eval/ct/ct_phase_tuning_gauss_mixing.py \
--clip-path clip/ViT-B-16.pt \
--num_workers 16 \
--config-file dinov2/configs/eval/clip_b16.yaml \
--pretrained-weights $PRETRAINED_WEIGHTS \
--output-dir ct_eval/ \
--category dtd \
--template-type simple+cafo --is_test True \
--eval_only True \
--temperature $TEMPERATURE \
--classifier_path $CLASSIFER \
--adapter_path $ADAPTER \
--text_path $TEXT \
--fuse_weight $FUSE_WEIGTH

