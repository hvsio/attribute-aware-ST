from dataclasses import dataclass
from typing import List, Dict
from sacrebleu.metrics import BLEU
import evaluate
from datasets import load_dataset, load_from_disk
from transformers import MBartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MBart50Tokenizer, AutoTokenizer,  DataCollatorForSeq2Seq, DefaultDataCollator, MBartTokenizer
from nltk.translate.bleu_score import corpus_bleu
from transformers.optimization import (
    Adafactor,
    get_linear_schedule_with_warmup,
)
import torch
import wandb
import os
import numpy as np
import nltk

os.environ["WANDB_PROJECT"] = "MT"
nltk.download("punkt")
LANG = "en"
MODEL_ID = 'facebook/mbart-large-cc25'
PATH = "/mnt/osmanthus/aklharas/speechBSD/transformers"
LOCAL = "/mnt/osmanthus/aklharas/checkpoints/mt2/checkpoint-700"
torch.cuda.empty_cache()

def get_model():
    model = MBartForConditionalGeneration.from_pretrained(MODEL_ID)
    tokenizer = MBartTokenizer.from_pretrained(MODEL_ID, tgt_lang="ja_XX", src_lang="en_XX")
    return model, tokenizer


def generate_datasets():
    model, tokenizer = get_model()
    sets = ['train', 'test', 'validation']

    for set_name in sets:
        print("1. Loading custom files")
        json_ds = load_dataset("json", data_files=f"/mnt/osmanthus/aklharas/speechBSD/transformers/jsons/{set_name}.json")
        print("2. Creating custom uncached files")
        dataset = json_ds.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer})
        print(dataset['train'][0])
        print("3. Saving to disk")
        dataset.save_to_disk(f"/mnt/osmanthus/aklharas/speechBSD/transformers/mt3/{set_name}_{LANG}.data")

source_lang = "en"
target_lang = "ja"
metric = evaluate.load("sacrebleu", tokenize="ja-mecab")
def preprocess_function(examples, tokenizer):
    inputs = examples[f"{source_lang}_sentence"]
    targets = examples[f"{target_lang}_sentence"]
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ja_XX"
    models_inputs = tokenizer(inputs, text_target=targets, return_tensors="pt", truncation=True, padding=True, max_length=128)
    return {k: torch.squeeze(v, 0) for k,v in models_inputs.items()}


def run(train=False, test=False, eval=False):
    wandb.init()
    model, tokenizer = get_model()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')
    model.to(device)
    print(f"CUDA available: {cuda}, device to {device}")
    print(next(model.parameters()).device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    #optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-5)
    #lr_scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                           num_training_steps=37500,
    #                                           num_warmup_steps=500)

    train_ds = load_from_disk(f"{PATH}/mt2/train_en.data/train")
    validation_ds = load_from_disk(f"{PATH}/mt2/validation_en.data/train")
#    validation_ds = validation_ds.select(range(20))
    test_ds = load_from_disk(f"{PATH}/mt2/test_en.data/train")

    def compute_metrics(pred):
     preds, labels = pred

     # decode preds and labels
     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
     decoded_labels = [[l] for l in decoded_labels]
     print(decoded_labels)
     print(decoded_preds)
     bleu = BLEU(tokenize="ja-mecab")
     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
     sacrebleu_score = bleu.corpus_score(decoded_preds, decoded_labels)
     print(result)
     print(sacrebleu_score)
     for i in range(20):
       print(f"{decoded_preds[i]} ---- {decoded_labels[i]}")
     wandb.log({"bleu": sacrebleu_score.score})
     return result

    train_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/osmanthus/aklharas/checkpoints/mt3cc",
        evaluation_strategy="steps",
        predict_with_generate=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        load_best_model_at_end=True,
        eval_accumulation_steps=2,
        gradient_accumulation_steps=4,
        num_train_epochs=30,
        bf16=True,
        save_steps=700,
        eval_steps=700,
        logging_steps=700,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=10,
        report_to=["wandb"],
        metric_for_best_model="eval_loss",
        optim="adafactor"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=data_collator,
    )

    if train:
        trainer.train()
        #trainer.train(resume_from_checkpoint="/mnt/osmanthus/aklharas/checkpoints/mt/checkpoint-2000")
        tokenizer.save_pretrained('/mnt/osmanthus/aklharas/models/mt2newdataset')
        trainer.save_model('/mnt/osmanthus/aklharas/models/mt2newdataset')
    elif eval:
       trainer.evaluate()
    elif test:
       trainer.predict(test_ds)

if __name__ == "__main__":
    #generate_datasets()
    run(eval=True)

