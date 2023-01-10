import os
import random

import torch
import torchaudio
from datasets import load_dataset
from transformers import logging, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer

from hf_model import HFSpeechMixEEDmBart

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class DataLoader:

    def __init__(self, model, cache, path, with_tag_g: False, with_tag_r: False, with_tags: False):
        self.wav = "facebook/wav2vec2-large-960h-lv60-self"
        self.wavTokenizer = Wav2Vec2Tokenizer.from_pretrained(self.wav)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav)
        self.processor = Wav2Vec2Processor(self.feature_extractor, self.wavTokenizer)
        self.model = model
        self.cache = cache
        self.path = path
        self.with_tag_g = with_tag_g
        self.with_tag_r = with_tag_r
        self.with_tags = with_tags

    def _create_self_decoder_input(self, tokenizer, input_sent, target_sent, device, gender=False, region=False, both=False):
        rnd = int(random.uniform(1, 10)) % 9 == 0
        gen_input = tokenizer(input_sent, add_special_tokens=True, return_tensors="pt").input_ids
        predicted = tokenizer(target_sent, add_special_tokens=True, return_tensors="pt").input_ids
        tag = gender or region or both
        print(gen_input)
        if tag:
            gen_input[0] = tag
        gen_input = torch.tensor(gen_input)
        print(gen_input)
        return gen_input, predicted[0]

    def _prepare_dataset_custom(self, batch, input_text_prompt="", selftype=False, split_type="train", lang="en"):
        region = "prefecture" if lang == "ja" else "state"
        tag_g = False
        tag_r = False
        tag_both = False

        if self.with_tag_g:
            tag_g = "<" + batch[f"{lang}_spk_gender"] + ">"
        elif self.with_tag_r:
            tag_r = "<" + batch[f"{lang}_spk_{region}"] + ">"
        elif self.with_tags:
            tag_both = "<" + batch[f"{lang}_spk_{region}"] + ">" + "<" + batch[f"{lang}_spk_gender"] + ">"
        print(tag_g, tag_r, tag_both)

        filename = batch[f"{lang}_wav"]
        speech, sampling_rate = torchaudio.load(f"{self.path}/wav/{split_type}/{filename}")
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        input_values = self.processor(resampler(speech).squeeze().numpy(), sampling_rate=16_000).input_values[0]
        batch["input_values"] = input_values

        batch["lengths"] = len(batch["input_values"])
        source_sent = batch[f"{lang}_sentence"]
        target_sent = batch[f"{'ja' if lang == 'en' else 'en'}_sentence"]

        decoder_input, decoder_target = self._create_self_decoder_input(self.model.tokenizer,
                                                                        input_text_prompt + source_sent,
                                                                        target_sent,
                                                                        next(self.model.parameters()).device,
                                                                        gender=tag_g,
                                                                        region=tag_r,
                                                                        both=tag_both
                                                                        )
        print(decoder_input, decoder_target)
        batch["input_text_prompt"] = input_text_prompt
        batch["text_input_ids"] = decoder_input
        batch["labels"] = decoder_target

        return batch


    def load_custom_datasets(self, set_name, lang, cache, note):
        selftype = False
        dataset = None

        if cache and os.path.isdir(self.path):
            logger.info("Getting cached files")
            #dataset = load_from_disk(f"{self.path}/transformers/{set_name}_{next(self.model.parameters()).device}_{lang}_{note}.data")
        else:
            logger.info("1. Loading custom files")
            json_ds = load_dataset("json", data_files=f"{self.path}/transformers/jsons/{set_name}.json", cache_dir="./.cache")
            logger.info("2. Creating custom uncached files")
            dataset = json_ds.map(self._prepare_dataset_custom,
                                  fn_kwargs={"selftype": selftype, "input_text_prompt": "", "split_type": f"{set_name}",
                                             "lang": lang})
            logger.info("3. Saving to disk")
            dataset.save_to_disk(f"{self.path}/transformers/{note}/{set_name}_{next(self.model.parameters()).device}_{lang}_{note}.data")
        return dataset


WAV = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
MBART = "facebook/mbart-large-50-many-to-many-mmt"

input_args = {'speech_model_config': WAV, 'nlp_model_config': MBART, 'SpeechMixEED': False,
              'SpeechMixED': False, 'SpeechMixSelf': False, 'SpeechMixAdapter': False, 'SpeechMixGAN': False,
              'SpeechMixFixed': False, 'HFSpeechMixEED': True, 'HFSpeechMixED': False, 'HFSpeechMixSelf': False,
              'HFSpeechMixAdapter': False, 'HFSpeechMixGAN': False, 'HFSpeechMixFixed': False, 'cache': False,
              'field': 'clean', 'train_split': 'train.100', 'test_split': 'validation', 'notes': 'base',
              'grad_accum': 20, 'logging_steps': 10, 'warmup_steps': 500, 'unfreeze_warmup_steps': 1000,
              'save_total_limit': 2, 'max_grad_norm': 10, 'worker': 15, 'batch': 3, 'epoch': 30, 'lr': 4e-05,
              'eval_step': 700, 'share_layer_ratio': 0.5, 'down_scale': 2, 'weighted_sum': False,
              'fixed_parameters': False, 'custom_set_path': 'speechBSD', 'max_input_length_in_sec': 20,
              'group_by_length': False,
              'fixed_except': ['layer_norm', 'encoder_attn', 'enc_to_dec_proj', 'length_adapter', 'layernorm_embedding',
                               'attention', 'encoder'], 'fp16': False, 'wandb': True}

model_type = "HFSpeechMixEEDmBart"
model = HFSpeechMixEEDmBart(**input_args)
device = torch.device("cuda")
model.to(device)
print(next(model.parameters()).device)
dl = DataLoader(model, False, "/mnt/osmanthus/aklharas/speechBSD", with_tag_g=True, with_tag_r=False, with_tags=False)
sets = ['validation', 'test', 'train']
for i in sets:
    dl.load_custom_datasets(i, "en", False, "en_gender")
