python train.py --speech_model_config "facebook/wav2vec2-base" \
--nlp_model_config "facebook/mbart-large-50-many-to-many-mmt" \
--custom_set_path "speechBSD" \
--batch 1 \
--grad_accum 16 \
--epoch 30 \
--worker 10 \
--share_layer_ratio 0.5 \
--down_scale 8 \
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
--local "tunedBoth8scale" \
--checkpoint "checkpoint-2100" \
--eval True

#recent train
accelerate launch train.py --speech_model_config jonatasgrosman/wav2vec2-large-xlsr-53-english --nlp_model_config facebook/mbart-large-50-many-to-many-mmt --custom_set_path en_gender --batch 2 --grad_accum 8 --epoch 30 --worker 10 --share_layer_ratio 0.5 --down_scale 8 --lr 4e-5 --warmup_steps 500 --modelpath en_gender --fixed_parameters True

#recent test
accelerate launch train.py --speech_model_config jonatasgrosman/wav2vec2-large-xlsr-53-english --nlp_model_config facebook/mbart-large-50-many-to-many-mmt --custom_set_path en_gender --batch 1 --grad_accum 16 --local en_gender --checkpoint checkpoint-18200 --test True

#sacrebleu
sacrebleu ref.txt -i hyp.txt -l en-ja -b 


