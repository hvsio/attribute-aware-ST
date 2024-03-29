import copy

import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, MBart50Tokenizer
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers import Wav2Vec2Model, MBartForConditionalGeneration
from transformers import logging

logger = logging.get_logger('train')
DATA_PATH = ""


def handle_decoder_input_none(decoder_config, batch=1, device=torch.device("cuda")):
    return torch.tensor([[decoder_config.decoder_start_token_id]] *
                        batch).to(device)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int,
                       decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class HFSpeechMixEEDmBart(PreTrainedModel):
    main_input_name = "input_values"

    def __init__(
            self,
            speech_model_config=None,
            nlp_model_config=None,
            share_layer_ratio=0.5,
            down_scale=8,
            weighted_sum=False,
            fixed_parameters=True,
            fixed_except=[
                "layer_norm",
                "encoder_attn",
                "enc_to_dec_proj",
                "length_adapter",
                "layernorm_embedding",
                "attention"
            ],
            gender_tags=False,
            en_tags=False,
            ja_tags=False,
            **kwargs,
    ):
        if isinstance(speech_model_config, PretrainedConfig):
            config = speech_model_config
            speech_model_config = config.to_dict().get('encoder')['_name_or_path']
            nlp_model_config = config.to_dict().get('decoder')['_name_or_path']
        else:
            config = SpeechMixConfig.from_configs(speech_model_config, nlp_model_config)

        super(HFSpeechMixEEDmBart, self).__init__(config)
        source_lang = "ja_XX"
        target_lang = "en_XX"
        additional_tokens = False

        if gender_tags or en_tags or ja_tags:
            additional_tokens = True
            self.tokenizer = MBart50Tokenizer.from_pretrained(f"{DATA_PATH}/tag_tokenizers/ja", src_lang=source_lang,
                                                              tgt_lang=target_lang)
            print("Added tokens")
        else:
            self.tokenizer = MBart50Tokenizer.from_pretrained(nlp_model_config, src_lang=source_lang,
                                                              tgt_lang=target_lang)

        self.encoder_model = Wav2Vec2Model.from_pretrained(speech_model_config)
        self.decoder_model = MBartForConditionalGeneration.from_pretrained(nlp_model_config)
        if additional_tokens:
            self.decoder_model.resize_token_embeddings(len(self.tokenizer))
            assert self.decoder_model.vocab_size == len(self.tokenizer)
        else:
            print("No tokens added")
        self.weighted_sum = weighted_sum
        num_nlp_encoder_layers = 0

        if hasattr(self.decoder_model.base_model.encoder, "layers"):
            num_nlp_encoder_layers = len(
                self.decoder_model.base_model.encoder.layers)
        elif hasattr(self.decoder_model.base_model.encoder, "block"):
            num_nlp_encoder_layers = len(
                self.decoder_model.base_model.encoder.block)

        print(
            "Before layer sharing num_speech_encoder_layers",
            len(self.encoder_model.encoder.layers),
        )

        remove_layers = (int(
            len(self.encoder_model.encoder.layers) *
            share_layer_ratio) if share_layer_ratio != 0 else 0)

        self.encoder_model.encoder.layers = self.encoder_model.encoder.layers[:len(
            self.encoder_model.encoder.layers) - remove_layers]

        self.num_speech_encoder_layers = len(self.encoder_model.encoder.layers)

        print(
            "After layer sharing ",
            "num_speech_encoder_layers",
            len(self.encoder_model.encoder.layers),
            "num_nlp_encoder_layers",
            num_nlp_encoder_layers,
            "share_layer_ratio",
            share_layer_ratio,
            "remove_layers",
            remove_layers,
        )

        # Downsample
        self.downsize = down_scale
        self.downloop = int(math.log(self.downsize, 2))
        if self.downsize > 1:
            self.length_adapters = nn.Sequential(*[
                nn.Conv1d(
                    in_channels=self.encoder_model.config.hidden_size,
                    out_channels=self.encoder_model.config.hidden_size,
                    kernel_size=2,
                    stride=2,
                ) for _ in range(self.downloop)
            ])
        else:
            self.length_adapters = nn.Sequential(nn.Identity())

        if self.weighted_sum:
            self.weights_sum = nn.Parameter(
                torch.zeros(self.num_speech_encoder_layers + 1))
        self.enc_to_dec_proj = nn.Linear(self.encoder_model.config.hidden_size,
                                         self.decoder_model.config.hidden_size)
        self.custom_modules(**kwargs)
        if fixed_parameters:
            self.encoder_model.eval()
            self.decoder_model.eval()
            for xcoder in [
                self.encoder_model.named_parameters,
                self.decoder_model.named_parameters,
            ]:
                for name, param in xcoder():
                    if param.requires_grad:
                        if any([k in name for k in fixed_except]):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

        list_no_grad = []
        list_grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                list_grad.append(name)
            else:
                list_no_grad.append(name)

        self.nlp_emb = self.decoder_model.get_input_embeddings()
        self.speech_encoder_layer = len(self.encoder_model.encoder.layers)
        self.nlp_encoder_layer = num_nlp_encoder_layers
        self.list_grad = list_grad
        self.list_no_grad = list_no_grad

        self.decoder_outputs = None

    def get_encoder(self):
        return self.encoder_model

    def get_decoder(self):
        return self.decoder_model

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Conv1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id,
                                  self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # "attention_mask": attention_mask,
        # decoder_inputs = self.decoder_model.prepare_inputs_for_generation(input_ids, past=past)
        # decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # "decoder_attention_mask": decoder_attention_mask,
        input_dict = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "past_key_values": past,
            "decoder_input_ids": input_ids,
            # "decoder_attention_mask": decoder_attention_mask
        }
        input_dict.update(kwargs)
        return input_dict

    def _reorder_cache(self, past, beam_idx):
        return self.decoder_model._reorder_cache(past, beam_idx)

    def custom_modules(self, **kwargs):
        return None

    def cal_loss(
            self,
            inputs_embeds=None,
            text_input_ids=None,
            attention_mask=None,
            decoder_outputs=None,
            decoder_input_ids=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
    ):
        if past_key_values is None:
            self.decoder_outputs = None
        if inputs_embeds is not None:
            output = self.decoder_model(
                inputs_embeds=inputs_embeds,
                encoder_outputs=decoder_outputs if decoder_outputs else self.decoder_outputs,
                # sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
                attention_mask=attention_mask,  # to avoid performing attention on padding token indices.
                decoder_input_ids=decoder_input_ids,  # Indices of decoder input sequence tokens in the vocabulary
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        elif text_input_ids is not None:
            output = self.decoder_model(
                input_ids=text_input_ids,  # Indices of input sequence tokens in the vocabulary
                encoder_outputs=decoder_outputs if decoder_outputs else self.decoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        self.decoder_outputs = [output.encoder_last_hidden_state]
        return output

    def forward(
            self,
            input_values=None,  # audio inputs
            decoder_text_prompt=None,
            text_input_ids=None,  # source transciption
            decoder_input_ids=None,  # target transcription
            labels=None,
            encoder_outputs=None,
            decoder_outputs=None,
            past_key_values=None,
            use_cache=None,
            return_model_detail=True,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs,
    ):
        # input_values, text_input_ids, decoder_input_ids, labels,
        if encoder_outputs is None:
            # last_hidden_state, extract_features, hidden_states, attentions (None)
            encoder_outputs = self.encoder_model(input_values, output_hidden_states=True)
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(
                self.decoder_model.config,
                encoder_outputs.last_hidden_state.shape[0],
                device=self.device,
            )
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = shift_tokens_right(
                labels,
                self.decoder_model.config.pad_token_id,
                self.decoder_model.config.decoder_start_token_id,
            )
            # decoder_input_ids = labels
        inputs_embeds = encoder_outputs.last_hidden_state.to(self.device)

        if self.weighted_sum:
            # weighted sum
            stacked_feature = torch.stack(encoder_outputs["hidden_states"],
                                          dim=0)
            _, *origin_shape = stacked_feature.shape
            stacked_feature = stacked_feature.view(
                self.num_speech_encoder_layers + 1, -1)
            norm_weights = F.softmax(self.weights_sum, dim=-1)
            weighted_feature = (norm_weights.unsqueeze(-1) *
                                stacked_feature).sum(dim=0)
            inputs_embeds = weighted_feature.view(*origin_shape)

        inputs_embeds = self.length_adapters(inputs_embeds.transpose(1, 2)).transpose(1, 2)

        inputs_embeds = self.enc_to_dec_proj(inputs_embeds)

        if decoder_text_prompt is not None:
            text_prompt = self.nlp_emb(
                self.tokenizer(decoder_text_prompt, return_tensors="pt")["input_ids"].to(self.device))
            inputs_embeds = torch.cat((text_prompt.expand(inputs_embeds.shape[0], -1, -1), inputs_embeds), 1)
        if decoder_text_prompt is not None and text_input_ids is not None:
            assert ("Can only take str or input_ids as input, but not both")
        outputs = self.cal_loss(
            inputs_embeds=inputs_embeds,
            decoder_outputs=decoder_outputs,
            text_input_ids=text_input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        outputs["logits"] = torch.argmax(outputs["logits"], -1)
        return outputs
        # Seq2SeqLMOutput ->
        #   logits, past_key_value, , encoder_last_hidden state
        #   (empty) decoder_hidden_state, decoder_attentions, cross_attentions, encoder_hidden_states, encoder_attentionss


class SpeechMixConfig(PretrainedConfig):
    model_type = "speechmix"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
                "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type,
                                            **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type,
                                            **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_configs(
            cls,
            encoder_config: PretrainedConfig,
            decoder_config: PretrainedConfig,
            **kwargs,
    ) -> PretrainedConfig:
        encoder_config = AutoConfig.from_pretrained(encoder_config)
        decoder_config = AutoConfig.from_pretrained(decoder_config)

        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(),
                   decoder=decoder_config.to_dict(),
                   **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
