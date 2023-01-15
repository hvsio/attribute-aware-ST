import sys
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import evaluate
import asrp
import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, TrainerCallback, \
    TrainerState, TrainerControl, logging
from transformers.optimization import (
    Adafactor,
    get_linear_schedule_with_warmup,
)
from utils.config_parser import parse_args
from hf_model import HFSpeechMixEEDmBart, SpeechMixConfig
from datetime import datetime
import wandb
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from sacrebleu.metrics import BLEU
import os
os.environ["WANDB_PROJECT"] = "attribute-aware-ST"
logging.set_verbosity_info()
logger = logging.get_logger("trainer")


@dataclass
class DataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    selftype: bool = False

    # text_input_ids, input_values, labels
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch['input_values'] = pad_sequence([torch.tensor(feature["input_values"]) for feature in features],
                                             batch_first=True, padding_value=-100)

        label_features = [{"input_ids": feature['labels']} for feature in features]
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=True,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        if 'text_input_ids' in features[0]:
            text_features = [{"input_ids": feature['text_input_ids'][0]} for feature in features]
            text_batch = self.tokenizer.pad(
                text_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
            batch['text_input_ids'] = text_batch['input_ids']

        labels_batch = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyway
        if self.tokenizer.bos_token_id and (labels_batch[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels_batch = labels_batch[:, 1:]

        batch['labels'] = labels_batch
        print(batch['labels'])

        torch.cuda.empty_cache()
        return batch


def get_model(input_args, local='', checkpoint='', encdec=''):
    if local:
        print(f"loading checkpoint {local}")
        config = SpeechMixConfig.from_json_file(f'/mnt/osmanthus/aklharas/checkpoints/{local}/{checkpoint}/config.json')
        checkpoint = torch.load(f'/mnt/osmanthus/aklharas/checkpoints/{local}/{checkpoint}/pytorch_model.bin')
        #model = HFSpeechMixEEDmBart.from_pretrained("/mnt/osmanthus/aklharas/checkpoints/tunedBothAda32/checkpoint-7000", config=config)
        #model = HFSpeechMixEEDmBart(config, load_checkpoint=True, model_path=local, encdec_path=encdec)
        model = HFSpeechMixEEDmBart(config)
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = HFSpeechMixEEDmBart(**input_args)
    model_type = "HFSpeechMixEEDmBart"
    return model, model_type


def main(arg=None):
    wandb.init()
    input_args, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    print("input_args", input_args)

    model, model_type = get_model(input_args, input_args.get('local'), input_args.get('checkpoint'), input_args.get('encdec'))
    selftype = 'SpeechMixSelf' in model_type
    if __name__ == '__main__':
        cuda = torch.cuda.is_available()
        device = torch.device('cuda', 0) if cuda else torch.device('cpu')
        model.to(device)
        print(f"CUDA available: {cuda}, device to {device}")
        print(next(model.parameters()).device)

    def compute_metrics(pred):
        # predictions, label_ids, inputs (empty?)
        print("computing metricsss")
        pred_ids = pred.predictions[0]
        pred_ids = [i[i != -100] for i in pred_ids]
        pred_str = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        label_str = model.tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
        bleu = BLEU(tokenize="ja-mecab")
        #sacrebleu = evaluate.load("sacrebleu")
        #bleu_score = sacrebleu.compute(predictions=pred_str, references=gold_sentences, tokenize='ja-mecab')
        gold_sentences = [[l] for l in label_str]
        result = bleu.corpus_score(pred_str, gold_sentences)
        nltk_bleu_score = corpus_bleu(gold_sentences, pred_str)
        print(nltk_bleu_score)
        #path = f"/mnt/osmanthus/aklharas/checkpoints/{input_args.get('modelpath')}/pretrained_weights"
        #if not os.path.exists(path):
        #    os.makedirs(path)

        #new_weights_files = str(datetime.now())
        #path = path+"/"+new_weights_files
        #pathE = path+"/encoder"
        #pathD = path+"/decoder"
        #os.makedirs(path)
        #os.makedirs(pathE)
        #os.makedirs(pathD)

        #model.encoder_model.save_pretrained(pathE)
        #model.decoder_model.save_pretrained(pathD)

        # for l, p in zip(label_str, pred_str):
        #     print(l, "======", p)
        cer = asrp.cer(label_str, pred_str)
        wer = asrp.wer(label_str, pred_str)
        print("PRED vs GOLD")
        for i in range(20):
            print(f"{pred_str[i]}   ---   {label_str[i]}")
        print({"cer": cer, "wer": wer, "sacrebleu": result.score})
        wandb.log({ "cer": cer, "wer": wer, "sacrebleu": result.score})
        return {"cer": cer, "wer": wer, "sacrebleu": result.score}

    class FreezingCallback(TrainerCallback):
        def __init__(self, trainer, freeze_model, freeze_epoch=3):
            self.trainer = trainer
            self.freeze_model = freeze_model
            self.freeze_epoch = freeze_epoch
            self.current_step_idx = 0
            self.default_param_fix = {}
            self.name_list = []
            for name, param in self.freeze_model.named_parameters():
                self.name_list.append(name)
                self.default_param_fix[name] = param.requires_grad
            self.freeze_layers = int(len(self.default_param_fix.keys()) / freeze_epoch)

        def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            print("Freezing callback executed")
            if state.epoch < self.freeze_epoch:
                release = self.name_list[-int(self.freeze_layers * state.epoch):]
                for name, param in self.freeze_model.named_parameters():
                    if name in release:
                        param.requires_grad = self.default_param_fix[name]
                    else:
                        param.requires_grad = False
            else:
                for name, param in self.freeze_model.named_parameters():
                    param.requires_grad = self.default_param_fix[name]
            self.current_step_idx += 1

        def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            for name, param in self.trainer.model.named_parameters():
                param.requires_grad = True

    if 'custom_set_path' in input_args:
        print('load datasets')
        train_ds = load_from_disk(
            f"/mnt/osmanthus/aklharas/{input_args['custom_set_path']}/transformers/en_plain_mbartdoc/train_en_en_plain_mbartdoc.data/train")
        dev_ds = load_from_disk(f"/mnt/osmanthus/aklharas/{input_args['custom_set_path']}/transformers/en_plain_mbartdoc/validation_en_en_plain_mbartdoc.data/train")
        test_ds = load_from_disk(f"/mnt/osmanthus/aklharas/{input_args['custom_set_path']}/transformers/en_plain_mbartdoc/test_en_en_plain_mbartdoc.data/train")
        print('datasets loaded')
        train_ds = train_ds.remove_columns(['no', 'ja_speaker', 'en_sentence', 'ja_sentence', 'ja_spkid', 'en_spkid', 'ja_wav', 'en_wav', 'ja_spk_gender', 'en_spk_gender', 'ja_spk_prefecture', 'en_spk_state'])
        dev_ds = dev_ds.remove_columns(['no', 'ja_speaker', 'en_sentence', 'ja_sentence', 'ja_spkid', 'en_spkid', 'ja_wav', 'en_wav', 'ja_spk_gender', 'en_spk_gender', 'ja_spk_prefecture', 'en_spk_state'])
        test_ds = test_ds.remove_columns(['no', 'ja_speaker', 'en_sentence', 'ja_sentence', 'ja_spkid', 'en_spkid', 'ja_wav', 'en_wav','ja_spk_gender', 'en_spk_gender', 'ja_spk_prefecture', 'en_spk_state'])

    steps = (20000/(2*int(input_args['batch']))*input_args.get('epoch', 10))
    print(steps)
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-5)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_training_steps=steps,
                                               num_warmup_steps=500)
    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer, padding=True, selftype=selftype)
    temp_id = now = datetime.now()

    training_args = TrainingArguments(
        output_dir=f"/mnt/osmanthus/aklharas/checkpoints/{input_args.get('modelpath', temp_id.strftime('%d/%m/%Y-%H.%M'))}",
        resume_from_checkpoint=True,
        per_device_train_batch_size=int(input_args['batch']),
        per_device_eval_batch_size=int(input_args['batch']),
        gradient_accumulation_steps=int(input_args['grad_accum']),
        eval_accumulation_steps=16,
        group_by_length=input_args["group_by_length"],
	evaluation_strategy="steps",
        load_best_model_at_end=True,
        #fp16=input_args.get('fp16', True),
        bf16=True,
        num_train_epochs=input_args.get('epoch', 10),
        save_steps=input_args.get('eval_step', 700),
        eval_steps=input_args.get('eval_step', 3),
        logging_steps=input_args.get('logging_steps', 5),
        #learning_rate=input_args.get('lr', 1e-4),
        #warmup_steps=input_args.get('warmup_steps', 500),
        save_total_limit=input_args.get('save_total_limit', 2),
        dataloader_num_workers=input_args.get('worker', 5),
        report_to=["wandb"],
    )
    # some trainer problem - save all logistics on compute_metrics, cause out of memory, fix:argmax first;
    # dynamic padding on past key value, cause index error, fix: return only loss and logist
    # group_by_length took lots of time during preprocessing
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    )

    # https://discuss.huggingface.co/t/gradual-layer-freezing/3381/4
    freezing_callback = FreezingCallback(trainer, model.encoder_model, input_args.get('unfreeze_warmup_steps', 500))
    trainer.add_callback(freezing_callback)
    print('training!')

    if input_args.get('eval', False):
        trainer.evaluate()
    elif input_args.get('test', False):
        with torch.no_grad():
          test_ds = test_ds.select(range(100))
          res = trainer.predict(test_ds)
          wandb.log({"test result": res})
    else:
        trainer.train()
        #trainer.train(resume_from_checkpoint="/mnt/osmanthus/aklharas/checkpoints/tunedBothAda32/checkpoint-2800")
        trainer.save_model(f"/mnt/osmanthus/aklharas/models/{input_args.get('modelpath')}")


if __name__ == "__main__":
    main()
