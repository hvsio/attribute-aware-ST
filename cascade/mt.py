from dataclasses import dataclass
from typing import List, Dict

from transformers import MBartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, MBart50Tokenizer
from nltk.translate.bleu_score import corpus_bleu
import torch
import wandb
import json
import os

os.environ["WANDB_PROJECT"] = "MT"

LANG = "en"
MODEL_ID = 'facebook/mbart-large-50-many-to-one-mmt'
PATH = "/mnt/osmanthus/aklharas/speechBSD/transformers"


def get_model():
    model = BartForConditionalGeneration.from_pretrained(MODEL_ID)
    tokenizer = MBart50Tokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def generate_datasets():
    sets = ['train', 'test', 'validation']

    for set_name in sets:
        ds = []
        print("1. Loading custom files")
        f = open(f"{PATH}/{set_name}.json")
        json_ds = json.load(f)
        print("2. Creating custom uncached files")
        for batch in json_ds:
            ds.append({
                "translation": {
                    "ja": batch["ja_sentence"],
                    "en": batch["en_sentence"]
                }
            })
        print("3. Saving to disk")
        with open(f"{PATH}/mt/{set_name}_{LANG}.json", 'w') as fout:
            json.dump(ds, fout, ensure_ascii=False)


def run(train=False, test=False, eval=False):
    wandb.init()
    model, tokenizer = get_model()
    print(next(model.parameters()).device)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')
    model.to(device)
    print(f"CUDA available: {cuda}, device to {device}")
    print(next(model.parameters()).device)

    @dataclass
    class DataCollator:
        tokenizer: BartTokenizer

        def __call__(self, features: List) -> Dict[str, torch.Tensor]:
            labels = [f["translation"]["en"] for f in features]
            inputs = [f["translation"]["ja"] for f in features]

            batch = self.tokenizer.prepare_seq2seq_batch(src_texts=inputs, src_lang="en_XX", tgt_lang="ja_XX",
                                                         tgt_texts=labels, max_length=32, max_target_length=32)

            for k in batch:
                batch[k] = torch.tensor(batch[k])
            return batch

    data_collator = DataCollator(tokenizer)

    train_ds = open(f"{PATH}/mt/train_en.json")
    train_ds = json.load(train_ds)
    validation_ds = open(f"{PATH}/mt/validation_en.json")
    validation_ds = json.load(validation_ds)
    test_ds = open(f"{PATH}/mt/test_en.json")
    test_ds = json.load(test_ds)

    def compute_metrics(pred):
        # predictions, label_ids, inputs (empty?)
        pred_ids = pred.predictions[0]
        pred_ids = [i[i != -100] for i in pred_ids]
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

        gold_sentences = [[l] for l in label_str]

        nltk_bleu_score = corpus_bleu(gold_sentences, pred_str)
        print("PRED vs GOLD")
        for i in range(20):
            print(f"{pred_str[i]}   ---   {label_str[i]}")
        print({"bleu": nltk_bleu_score})
        wandb.log({"bleu": nltk_bleu_score})
        return {"bleu": nltk_bleu_score}

    train_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/osmanthus/aklharas/checkpoints/mt",
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=10,
        report_to=["wandb"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=train_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
    )

    if train:
        trainer.train()
        # trainer.train(resume_from_checkpoint="/mnt/osmanthus/aklharas/checkpoints/mt/checkpoint-6000")
        tokenizer.save_pretrained('/mnt/osmanthus/aklharas/models/mt')
        trainer.save_model('/mnt/osmanthus/aklharas/models/mt')
    elif eval:
        trainer.evaluate()
    elif test:
        with torch.no_grad():
            test_ds = test_ds.select(range(100))
            res = trainer.predict(test_ds)
            print(res)


if __name__ == "__main__":
    generate_datasets()
    run(train=True)
