python train.py --speech_model_config facebook/wav2vec2-large-960h-lv60-self \
--nlp_model_config facebook/mbart-large-50-many-to-many-mmt \
--custom_set_path "speechBSD" \
--batch 64 \
--grad_accum 20 \
--epoch 30 \
--worker 15 \
--share_layer_ratio 0.5 \
--down_scale 2 \
--lr 4e-5 \
--warmup_steps 500 \
--notes v1