from datasets import load_dataset, load_from_disk
import os
import torchaudio
import torch
from transformers import logging, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer, Wav2Vec2Processor
from hf_model import HFSpeechMixEEDmBart
import re

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class DataLoader:

    def __init__(self, model, cache, path):
        self.wav = "facebook/wav2vec2-large-960h-lv60-self"
        self.chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        self.wavTokenizer = Wav2Vec2Tokenizer.from_pretrained(self.wav)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav)
        self.processor = Wav2Vec2Processor(self.feature_extractor, self.wavTokenizer)
        self.model = model
        self.cache = cache
        self.path = path

    def _create_self_decoder_input(self, enja_tokenizer, jaen_tokenizer, input_sent, golden_sentence, lang):
        target_tokenizer = enja_tokenizer if lang == "en" else jaen_tokenizer
        source_tokenizer = enja_tokenizer if lang == "en" else jaen_tokenizer

        gen_input = source_tokenizer(input_sent, add_special_tokens=True, return_tensors="pt").input_ids
        predicted = target_tokenizer(golden_sentence, add_special_tokens=True, return_tensors="pt").input_ids

        return gen_input, predicted[1:]

    def _prepare_dataset_custom(self, batch, input_text_prompt="", selftype=False, split_type="train", lang="en"):
        filename = batch[f"{lang}_wav"]
        target_lang = "ja" if lang == "en" else "en"
        speech, sampling_rate = torchaudio.load(f"{self.path}/wav/{split_type}/{filename}")
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        input_values = self.processor(resampler(speech).squeeze().numpy(), sampling_rate=16_000).input_values[0]
        batch["input_values"] = input_values

        batch["lengths"] = len(batch["input_values"])
        sent = re.sub(self.chars_to_ignore_regex, '', batch[f"{lang}_sentence"]).lower()
        golden_sentence = re.sub(self.chars_to_ignore_regex, '', batch[f"{target_lang}_sentence"]).lower()
        #sent = batch[f"{lang}_sentence"].lower()

        decoder_input, decoder_target = self._create_self_decoder_input(self.model.tokenizerENJA, self.model.tokenizerJAEN,
                                                                        input_text_prompt + sent,
                                                                        golden_sentence,
                                                                        lang)
        batch["input_text_prompt"] = input_text_prompt
        batch["text_input_ids"] = decoder_input
        batch["labels"] = decoder_target
        batch["labels"] += [self.model.tokenizer.eos_token_id]

        return batch


    def load_custom_datasets(self, set_name, lang, cache, note):
        selftype = False
        json_ds = None
        dataset = None

        if cache and os.path.isdir(self.path):
            logger.info("Getting cached files")
            dataset = load_from_disk(f"{self.path}/transformers/{set_name}_{next(self.model.parameters()).device}_{lang}_{note}.data")
        else:
            logger.info("1. Loading custom files")
            logger.info(f"{self.path}/transformers/{set_name}_{next(self.model.parameters()).device}_{lang}_{note}.data")
            json_ds = load_dataset("json", data_files=f"{self.path}/transformers/{set_name}.json", cache_dir="./.cache")
            logger.info("2. Creating custom uncached files")
            dataset = json_ds.map(self._prepare_dataset_custom,
                                  fn_kwargs={"selftype": selftype, "input_text_prompt": "", "split_type": f"{set_name}",
                                             "lang": lang})
            logger.info("3. Saving to disk")
            dataset.save_to_disk(f"{self.path}/transformers/{set_name}_{next(self.model.parameters()).device}_{lang}_{note}.data")
        return dataset

#v1
input_args = {'speech_model_config': 'facebook/wav2vec2-large-960h-lv60-self', 'nlp_model_config': 'facebook/mbart-large-50-many-to-many-mmt', 'SpeechMixEED': False,
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
dl = DataLoader(model, False, "speechBSD")
dl.load_custom_datasets("validation", "en", False, "HF_EED_mbart_fixedtarget")
