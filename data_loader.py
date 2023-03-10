import torch
import torchaudio
from datasets import load_dataset
from transformers import logging, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer, MBart50Tokenizer

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class DataLoader:

    def __init__(self, tokenizer, path, with_tag_g: False, with_tag_r: False):
        self.wav = "facebook/wav2vec2-large-960h-lv60-self"
        self.wavTokenizer = Wav2Vec2Tokenizer.from_pretrained(self.wav)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav)
        self.processor = Wav2Vec2Processor(self.feature_extractor, self.wavTokenizer)
        self.tokenizer = tokenizer
        self.path = path
        self.with_tag_g = with_tag_g
        self.with_tag_r = with_tag_r

    def _create_self_decoder_input(self, tokenizer, input_sent, target_sent, tag1=False, tag2=False):
        gen_input = tokenizer(input_sent, text_target=target_sent, add_special_tokens=True)
        inputs = gen_input.input_ids
        labels = gen_input.labels
        if tag1:
            tag1_id = tokenizer.convert_tokens_to_ids([tag1])
            #inputs = torch.cat([torch.tensor([[tag_id]]), inputs], dim=1)  #fix that
            inputs.insert(1, tag1_id[0]) #try also reverse order region + gender
            #labels.insert(1, tag1_id[0])
        if tag2:
            tag2_id = tokenizer.convert_tokens_to_ids([tag2])
            inputs.insert(1, tag2_id[0])
            #labels.insert(1, tag2_id[0])
        return torch.tensor([inputs], dtype=torch.int32), labels

    def _prepare_dataset_custom(self, batch, input_text_prompt="", split_type="train", lang="en"):
        region = "prefecture" if lang == "ja" else "state"

        tag1 = False
        tag2 = False
        if self.with_tag_g:
            tag1 = "<" + batch[f"{lang}_spk_gender"] + ">"
        if self.with_tag_r:
            #tag2 = "<" + batch[f"{lang}_spk_{region}"] + ">"
            tag2 = _convert_to_dialect_token(batch[f"{lang}_spk_state"])
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
                                                                        tag1=tag1,
                                                                        tag2=tag2,
                                                                        )
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
                              fn_kwargs={"input_text_prompt": "", "split_type": f"{set_name}",
                                         "lang": lang})
        logger.info("3. Saving to disk")
        dataset.save_to_disk(
            f"{self.path}/transformers/{note}/{set_name}.data")
        return dataset


def _convert_to_dialect_token(region):
    token = ''
    if region == 'MA':
        token = '<NWE>'
    elif region in ['IL', 'MI', 'IA', 'OH']:
        token = '<MID>'
    elif region in ['FL', 'GA', 'SC', 'KY', 'VA']:
        token = '<SOU>'
    elif region in ['CA', 'OR', 'CO']:
        token = '<WES>'
    else:
        print(f"Not found {region}")
        exit()
   # print(f"final token {region}")
    return token
def generate(tokenizer):
    #tokenizer = MBart50Tokenizer.from_pretrained("/mnt/osmanthus/aklharas/models/tag_tokenizers/en/gender")
    device = torch.device("cuda")
    dl = DataLoader(tokenizer, "/mnt/osmanthus/aklharas/speechBSD", with_tag_g=False,
                    with_tag_r=True)
    sets = ['validation', 'test', 'train']
    for i in sets:
        dl.load_custom_datasets(i, "en", "en_dialect_special")


def create_tokenizer(gender_tags=False, en_tags=False, ja_tags=False):
    MBART = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(MBART)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ja_XX"
    additional_tokens = []
    if gender_tags:
        print("Adding gender tags...")
        additional_tokens = ['<F>', '<M>']

    if en_tags:
        print("Adding EN region...")
        #additional_tokens = additional_tokens + ['<FL>', '<GA>', '<IA>', '<IL>', '<CO>', '<OH>', '<KY>', '<OR>',
        #                                         '<MI>', '<VA>', '<MA>', '<CA>', '<SC>']
        additional_tokens = additional_tokens + ['<NWE>', '<MID>', '<SOU>', '<WES>']
    elif ja_tags:
        print("Adding JA region...")
        additional_tokens = additional_tokens + ['<??????>', '<??????>', '<??????>', '<??????>', '<??????>', '<??????>',
                                                 '<??????>', '<??????>',
                                                 '<?????????>', '<??????>', '<??????>', '<??????>', '<??????>', '<??????>',
                                                 '<??????>', '<??????>',
                                                 '<??????>', '<??????>', '<??????>', '<??????>', '<??????>', '<??????>',
                                                 '<??????>', '<??????>',
                                                 '<??????>', '<??????>', '<??????>', '<??????>', '<?????????>', '<??????>']
    if additional_tokens:
      tok_list = tokenizer._additional_special_tokens
      tok_list =  additional_tokens + tok_list
      tok_dist = {'additional_special_tokens': tok_list }
      print(tok_list)
      #tokenizer.add_tokens(tok_list)
      tokenizer.add_special_tokens(tok_dist)
      tokenizer.save_pretrained("/mnt/osmanthus/aklharas/tag_tokenizers/en/dialect_special")
    return tokenizer

if __name__ == "__main__":
     create_tokenizer(en_tags=True)
     #tokenizer = create_tokenizer(gender_tags=True, en_tags=True)
     tokenizer = MBart50Tokenizer.from_pretrained("/mnt/osmanthus/aklharas/tag_tokenizers/en/dialect_special")
     tokenizer.src_lang = "en_XX"
     tokenizer.tgt_lang = "ja_XX"
     generate(tokenizer)
