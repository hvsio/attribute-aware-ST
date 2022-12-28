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
from utils.config_parser import parse_args
from hf_model import HFSpeechMixEEDmBart, SpeechMixConfig
from datetime import datetime
import wandb

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

        torch.cuda.empty_cache()
        return batch


def get_model(input_args, local=''):
    if local:
        print(f"loading checkpoint {local}")
        config = SpeechMixConfig.from_json_file(f'./{local}/config.json')
        checkpoint = torch.load(f'./{local}/pytorch_model.bin')
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

    model, model_type = get_model(input_args, input_args.get('local'))
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

        gold_sentences = [[l] for l in label_str]

        sacrebleu = evaluate.load("sacrebleu")
        bleu_score = sacrebleu.compute(predictions=pred_str, references=gold_sentences)

        # for l, p in zip(label_str, pred_str):
        #     print(l, "======", p)
        cer = asrp.cer(label_str, pred_str)
        wer = asrp.wer(label_str, pred_str)
        print("PRED vs GOLD")
        for i in range(20):
            print(f"{pred_str[i]}   ---   {label_str[i]}")
        print({"cer": cer, "wer": wer, "bleu": bleu_score['score']})
        wandb.log({ "cer": cer, "wer": wer, "bleu": bleu_score['score']})
        return {"cer": cer, "wer": wer, "bleu": bleu_score['score']}

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
            f"{input_args['custom_set_path']}/transformers/train_cuda:0_en_HF_EED_mbart_cuda.data/train")
        dev_ds = load_from_disk(f"{input_args['custom_set_path']}/transformers/validation_cuda:0_en_HF_EED_mbart_cuda.data/train")
        test_ds = load_from_disk(f"{input_args['custom_set_path']}/transformers/test_cuda:0_en_HF_EED_mbart_cuda.data/train")
        print('datasets loaded')
        train_ds = train_ds.remove_columns(['no', 'ja_speaker', 'en_sentence', 'ja_sentence', 'ja_spkid', 'en_spkid', 'ja_wav', 'en_wav', 'ja_spk_gender', 'en_spk_gender', 'ja_spk_prefecture', 'en_spk_state'])
        dev_ds = dev_ds.remove_columns(['no', 'ja_speaker', 'en_sentence', 'ja_sentence', 'ja_spkid', 'en_spkid', 'ja_wav', 'en_wav', 'ja_spk_gender', 'en_spk_gender', 'ja_spk_prefecture', 'en_spk_state'])
        test_ds = test_ds.remove_columns(['no', 'ja_speaker', 'en_sentence', 'ja_sentence', 'ja_spkid', 'en_spkid', 'ja_wav', 'en_wav','ja_spk_gender', 'en_spk_gender', 'ja_spk_prefecture', 'en_spk_state'])

    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer, padding=True, selftype=selftype)
    temp_id = now = datetime.now()

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{input_args.get('modelpath', temp_id.strftime('%d/%m/%Y-%H.%M'))}",
        per_device_train_batch_size=int(input_args['batch']),
        per_device_eval_batch_size=int(input_args['batch']),
        gradient_accumulation_steps=int(input_args['grad_accum']),
        eval_accumulation_steps=3,
        group_by_length=input_args["group_by_length"],
        optim="adafactor",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        fp16=input_args.get('fp16', True),
        num_train_epochs=input_args.get('epoch', 10),
        save_steps=input_args.get('eval_step', 700),
        eval_steps=input_args.get('eval_step', 3),
        logging_steps=input_args.get('logging_steps', 5),
        learning_rate=input_args.get('lr', 5e-4),
        warmup_steps=input_args.get('warmup_steps', 500),
        save_total_limit=input_args.get('save_total_limit', 2),
        dataloader_num_workers=input_args.get('worker', 10),
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
          print(res)
    else:
        trainer.train()
        trainer.save_model(f"./models/{input_args.get('modelpath')}")


if __name__ == "__main__":
    main()
