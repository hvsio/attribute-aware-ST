python train.py --speech_model_config "facebook/wav2vec2-base" \
--nlp_model_config "facebook/mbart-large-50-many-to-many-mmt" \
--custom_set_path "speechBSD" \
--batch 1 \
--grad_accum 16 \
--epoch 30 \
--worker 10 \
--share_layer_ratio 0.5 \
--down_scale 2 \
--lr 1e-5 \
--warmup_steps 500 \
--modelpath smallerWav \
--local checkpoints/checkpoint-11200

# eval checkpoint
train.py \
--custom_set_path "speechBSD" \
--batch 1 \
--epoch 10 \
--worker 10 \
--local "tunedBothAda32" \
--checkpoint "checkpoint-3500" \
--eval True
