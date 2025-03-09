export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=. 

OUTPUT_DIR=./output/clip_b16
torchrun --nnodes=1 --master_port=12345 --nproc_per_node=2 dinov2/train/train.py \
--config-file dinov2/configs/train/clip_b16.yaml \
--output-dir $OUTPUT_DIR \