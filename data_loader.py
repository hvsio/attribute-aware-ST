import torch
import torchaudio
from datasets import load_dataset
from transformers import logging, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer, MBart50Tokenizer

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class DataLoader:

    def __init__(self, tokenizer, path, with_tag_g: False, with_tag_r: False, with_tags: False):
        self.wav = "facebook/wav2vec2-large-960h-lv60-self"
        self.wavTokenizer = Wav2Vec2Tokenizer.from_pretrained(self.wav)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav)
        self.processor = Wav2Vec2Processor(self.feature_extractor, self.wavTokenizer)
        self.tokenizer = tokenizer
        self.path = path
        self.with_tag_g = with_tag_g
        self.with_tag_r = with_tag_r
        self.with_tags = with_tags

    def _create_self_decoder_input(self, tokenizer, input_sent, target_sent, tag=False):
        gen_input = tokenizer(input_sent, add_special_tokens=True, return_tensors="pt").input_ids
        predicted = tokenizer(target_sent, add_special_tokens=True, return_tensors="pt").input_ids
        print("BEFORE--------------------------")
        print(gen_input)
        if tag:
            tag_id = tokenizer.convert_token_to_id(tag)
            gen_input = torch.cat([torch.tensor([[tag_id]]), gen_input], dim=1)  # fix that, insert as second
        print("AFTER--------------------------")
        print(gen_input)
        return gen_input, predicted[0]

    def _prepare_dataset_custom(self, batch, input_text_prompt="", selftype=False, split_type="train", lang="en"):
        region = "prefecture" if lang == "ja" else "state"

        tag = False
        if self.with_tag_g:
            tag = "<" + batch[f"{lang}_spk_gender"] + ">"
        elif self.with_tag_r:
            tag = "<" + batch[f"{lang}_spk_{region}"] + ">"
        elif self.with_tags:
            tag = "<" + batch[f"{lang}_spk_{region}"] + ">" + "<" + batch[f"{lang}_spk_gender"] + ">"

        filename = batch[f"{lang}_wav"]
        speech, sampling_rate = torchaudio.load(f"{self.path}/wav/{split_type}/{filename}")
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        input_values = self.processor(resampler(speech).squeeze().numpy(), sampling_rate=16_000).input_values[0]
        batch["input_values"] = input_values

        batch["lengths"] = len(batch["input_values"])
        source_sent = batch[f"{lang}_sentence"]
        target_sent = batch[f"{'ja' if lang == 'en' else 'en'}_sentence"]

        decoder_input, decoder_target = self._create_self_decoder_input(self.tokenizer,
                                                                        input_text_prompt + source_sent,
                                                                        target_sent,
                                                                        tag=tag,
                                                                        )
        print(decoder_input, decoder_target)
        batch["input_text_prompt"] = input_text_prompt
        batch["text_input_ids"] = decoder_input
        batch["labels"] = decoder_target

        return batch

    def load_custom_datasets(self, set_name, lang, note):
        selftype = False
        dataset = None

        logger.info("1. Loading custom files")
        json_ds = load_dataset("json", data_files=f"{self.path}/transformers/jsons/{set_name}.json",
                               cache_dir="./.cache")
        logger.info("2. Creating custom uncached files")
        dataset = json_ds.map(self._prepare_dataset_custom,
                              fn_kwargs={"selftype": selftype, "input_text_prompt": "", "split_type": f"{set_name}",
                                         "lang": lang})
        logger.info("3. Saving to disk")
        dataset.save_to_disk(
            f"{self.path}/transformers/{note}/{set_name}_{next(self.model.parameters()).device}_{lang}_{note}.data")
        return dataset


def generate():
    tokenizer = MBart50Tokenizer.from_pretrained("/mnt/osmanthus/aklharas/models/tag_tokenizers/en/gender")
    device = torch.device("cuda")
    dl = DataLoader(tokenizer, "/mnt/osmanthus/aklharas/speechBSD/transformers", with_tag_g=True,
                    with_tag_r=False, with_tags=False)
    sets = ['validation', 'test', 'train']
    for i in sets:
        dl.load_custom_datasets(i, "en", "en_gender")


def create_tokenizer(gender_tags=False, en_tags=False, ja_tags=False):
    MBART = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(MBART)
    if gender_tags:
        print("Adding gender tags...")
        additional_tokens = ['<F>', '<M>']

    if en_tags:
        print("Adding EN region...")
        additional_tokens = additional_tokens + ['<FL>', '<GA>', '<IA>', '<IL>', '<CO>', '<OH>', '<KY>', '<OR>',
                                                 '<MI>', '<VA>', '<MA>', '<CA>', '<SC>']
    elif ja_tags:
        print("Adding JA region...")
        additional_tokens = additional_tokens + ['<沖縄>', '<岡山>', '<京都>', '<高知>', '<静岡>', '<栃木>',
                                                 '<茨城>', '<愛知>',
                                                 '<神奈川>', '<宮城>', '<秋田>', '<兵庫>', '<福岡>', '<千葉>',
                                                 '<熊本>', '<富山>',
                                                 '<岐阜>', '<群馬>', '<山梨>', '<香川>', '<不明>', '<滋賀>',
                                                 '<東京>', '<佐賀>',
                                                 '<新潟>', '<広島>', '<埼玉>', '<山形>', '<北海道>', '<大阪>']
    tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})
    tokenizer.save_pretrained("/mnt/osmanthus/aklharas/models/tag_tokenizers/en/gender")

if __name__ == "__main__":
    create_tokenizer(gender_tags=True)
    generate()