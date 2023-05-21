# Attribute Aware Speech Translation

This repository holds a setup for experiments with attribute aware E2E speech translation model. Attributes consists of speaker's gender, dialect or both combined in form of tags attached to the original utterance translation. 

## Environment
Experiments were conducted on 2xA100 40GB using python 3.9. 
Models used are based on transformers and highly rely on HuggingFace library.

For logging and training tracking WandB library was used. One can either configure it to point to their own WandB account or skip this part. 

## Data
Data used for training, evaluation and testing consists of business conversations in japanese and english with added attributes of gender and region/prefecture of the speaker. 

Note that paths used in the code for addressing the data, checkpoint and models themselves do not directly point to this repository contents. Those were stored outside of it.

Dataset can be found at [speechBSD corpus](https://github.com/ku-nlp/speechBSD).

## Models
For the E2E creation I have used a wav2vec encoder and mBart50 decoder with layer freezing strategy as inspired by [Multilingual Speech Translation
from Efficient Finetuning of Pretrained Models](https://aclanthology.org/2021.acl-long.68.pdf)

In order to maximize vibility of tag-insertion influence I have utilized additionally finetuned versions of both. Wav2Vec was fine tuned on either english or japanese and sourced from HF (wav2vec2-large-xlsr-53-english/-japanese), whereas mBart50 was fine tuned for multilingual machine translation (facebook/mbart-large-50-many-to-many-mmt). 

### Attributes
Tag attributes were added to the sentence at the beginning of the translated utterance, yet not interfering with the mBart50 language code requirement: [lang_code][gender|region|both tag] text [eos] as inspired by [Breeding Gender-aware Direct Speech Translation Systems”, Gaido et al., 2020](https://aclanthology.org/2020.coling-main.350.pdf).

Later on an attempt of aggregating similar dialects was made. Hence, data loader file contains a longer list of all dialects and a shorter one for dialects aggregated into larger, common regions. It was applied for both JA and EN (`_convert_to_en_dialect` and `_convert_to_jp_dialect`).


## Procedure
To replicate the experiment there are three main steps.

1. Convert the dataset to HF-compatible format. This is done in `data_loader.py` file executed as typical python file. Handling of adding attributes to the translations was done manually in the `main()` function.

    1.1 Data loader

2. Start training - pass parameters of your liking (described in `utils/config_parser.py`) as arguments when executing the `train.py` file. Two needed arguments not related to parameters are:
    2.1 `modelpath` - where the checkpoints are expected to be saved, if not provided it defaults to the timestamp of execution
    2.2 `custom_set_path` - points to the dataset used fot training/evaluation/testing

 Fx: `python train.py --speech_model_config jonatasgrosman/wav2vec2-large-xlsr-53-japanese --nlp_model_config facebook/mbart-large-50-many-to-many-mmt --custom_set_path ja_with_gender --batch 1 --grad_accum 16 --epoch 30 --worker 10 --share_layer_ratio 0.5 --down_scale 8 --lr 1e-5 --warmup_steps 500 --modelpath ja_gender --fixed_parameters True`


3. Test/evaluation
To test or evaluate trained model you need to provide used base models, batch details, dataset location and saved model config location (`local` as folder name and `checkpoint` as the version to test/eval). To force test or eval one needs to add either `--test` or `--eval` argument with `True` value.

Fx: `python train.py --speech_model_config jonatasgrosman/wav2vec2-large-xlsr-53-english --nlp_model_config facebook/mbart-large-50-many-to-many-mmt --custom_set_path en_notags --batch 1 --grad_accum 16 --local en_plain --checkpoint checkpoint-18200 --test True`

## Results
For the purpose of measuring the results sacrebleu metric is used, provided by HF. For JA use case, japanese tokenizer `ja-mecab` was used.
EN->JA
|  baseline | \<gender>   |  \<region>  |  \<dialect>  |  \<region+gender>  |
|:-:|:-:|:-:|:-:|:-:
| 15.5  | 15.6  | 15.3  | 15.7  | 15.2  |
|   |   |   |   |   |

JA->EN
|  baseline | \<gender>   |  \<region>  |  \<dialect>  |  \<region+gender>  |
|:-:|:-:|:-:|:-:|:-:
| 15.8  | 15.2  | 15.5  | 16.0  | 15.5  |
|   |   |   |   |   |

Across many tests there was a negligible difference between the baseline and the model train on datasets with attributes. One consistency seen across all of them was that <u>aggregated dialect tag</u> proven the most visible increase out of all other tags. Despite that, the increase is of small value ehnce no solid conclusion can be made without further investigation in that direction.

## Other
`accelerate` library was used for conducting expiments across multiple GPUs. 

This project was uptaken and supervised by Language Media Processing Lab at Kyoto University.

This repo is based on [SpeechMix](https://github.com/voidful/SpeechMix) with few personal modifications.