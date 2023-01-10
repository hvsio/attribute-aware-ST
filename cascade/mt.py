from dataclasses import dataclass
from typing import List, Dict

from datasets import load_dataset, load_from_disk
from transformers import MBartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MBart50Tokenizer, AutoTokenizer,  DataCollatorForSeq2Seq, DefaultDataCollator
from nltk.translate.bleu_score import corpus_bleu
from transformers.optimization import (
    Adafactor,
    get_linear_schedule_with_warmup,
)
import torch
import wandb
import os

os.environ["WANDB_PROJECT"] = "MT"

LANG = "en"
MODEL_ID = 'facebook/mbart-large-50-many-to-many-mmt'
PATH = "/mnt/osmanthus/aklharas/speechBSD/transformers"
LOCAL = "/mnt/osmanthus/aklharas/checkpoints/mt/checkpoint-1500"

def get_model():
    model = MBartForConditionalGeneration.from_pretrained(MODEL_ID)
    tokenizer = MBart50Tokenizer.from_pretrained(MODEL_ID, tgt_lang="ja_XX", src_lang="en_XX")
    return model, tokenizer


def generate_datasets():
    model, tokenizer = get_model()
    sets = ['train', 'test', 'validation']

    for set_name in sets:
        print("1. Loading custom files")
        json_ds = load_dataset("json", data_files=f"/mnt/osmanthus/aklharas/speechBSD/transformers/{set_name}.json")
        print("2. Creating custom uncached files")
        dataset = json_ds.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer})
        print(dataset['train'][0])
        print("3. Saving to disk")
        dataset.save_to_disk(f"/mnt/osmanthus/aklharas/speechBSD/transformers/mt/{set_name}_{LANG}.data")

source_lang = "en"
target_lang = "ja"

def preprocess_function(examples, tokenizer):
    inputs = examples[f"{source_lang}_sentence"]
    targets = examples[f"{target_lang}_sentence"]
    tokenizer.src_lang = "en_XX"
    models_inputs = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=128)
    labels = tokenizer(targets, return_tensors="pt", truncation=True, padding=True, max_length=128)
    models_inputs["labels"] = labels["input_ids"]
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
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-5)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_training_steps=37500,
                                               num_warmup_steps=500)

    train_ds = load_from_disk(f"{PATH}/mt/train_en.data/train")
    validation_ds = load_from_disk(f"{PATH}/mt/validation_en.data/train")
#    validation_ds = validation_ds.select(range(10))
    test_ds = load_from_disk(f"{PATH}/mt/test_en.data/train")

    def compute_metrics(pred):
        # predictions, label_ids, inputs (empty?)
        pred_ids = pred.predictions[0]
        pred_ids = pred_ids.argmax(-1)
        res = model.generate(*pred_ids, forced_bos_token_id=250012)
#        print(res)
        pred_ids = [i[i != -100] for i in pred_ids]
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        print(label_ids)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

        preds = [pred.strip() for pred in pred_str]
        labels = [[label.strip()] for label in label_str]
        print(labels)
        print(preds)

        nltk_bleu_score = corpus_bleu(labels, preds)
        print("PRED vs GOLD")
        for i in range(20):
            print(f"{pred_str[i]}   ---   {label_str[i]}")
        print({"bleu": nltk_bleu_score})
        wandb.log({"bleu": nltk_bleu_score})
        return {"bleu": nltk_bleu_score}

    train_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/osmanthus/aklharas/checkpoints/mt",
        evaluation_strategy="steps",
        predict_with_generate=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
        eval_accumulation_steps=16,
        gradient_accumulation_steps=4,
        num_train_epochs=30,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
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
#        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=data_collator,
    )

    if train:
        trainer.train()
        #trainer.train(resume_from_checkpoint="/mnt/osmanthus/aklharas/checkpoints/mt/checkpoint-2000")
        tokenizer.save_pretrained('/mnt/osmanthus/aklharas/models/mt1')
        trainer.save_model('/mnt/osmanthus/aklharas/models/mt1')
    elif eval:
        trainer.evaluate(metric_key_prefix="eval_bleu")
    elif test:
        with torch.no_grad():
            preds = []
            for i in range(21):
             print(f"Predicting {i} batch")
             #test_ds = test_ds.select(range(100))
             labels = [[l] for l in validation_ds["ja_sentence"]]
             #test_ds = test_ds["en_sentence"]
             end = (i+1)*100 if ((i+1)*100) <= len(validation_ds["en_sentence"]) else len(validation_ds["en_sentence"])
             print(end)
             tokenized = tokenizer(validation_ds["en_sentence"][i*100:end], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
             res = model.generate(**tokenized, forced_bos_token_id=250012)
             res = tokenizer.batch_decode(res, skip_special_tokens=True)
             preds = preds + res
             print(len(preds))
            for x in range(20):
              print(f"{res[x]} ------ {labels[x][0]}")
            nltk_bleu_score = corpus_bleu(labels, preds)
            print({"bleu": nltk_bleu_score})



if __name__ == "__main__":
    #generate_datasets()
    run(test=True)

