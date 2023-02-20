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
MODEL_ID = 'facebook/mbart-large-50-many-to-many-mmt'
LOCAL_ID = "/mnt/osmanthus/aklharas/tag_tokenizers/en/gender"
PATH = "/mnt/osmanthus/aklharas/speechBSD/transformers"
LOCAL = "/mnt/osmanthus/aklharas/checkpoints/mt6-50base/checkpoint-18200"
index = 0

def get_model(local=False):
    id = LOCAL if local else MODEL_ID
    print(f"loading {id} model")
    model = MBartForConditionalGeneration.from_pretrained(LOCAL)
    tokenizer = MBart50Tokenizer.from_pretrained(MODEL_ID, tgt_lang="ja_XX", src_lang="en_XX")
    #model.resize_token_embeddings(len(tokenizer))
    #assert model.vocab_size == len(tokenizer)
    print("passed")

    return model, tokenizer

def generate_cascade_testset():
   with open("asr.txt", "r") as f:
    samples = f.readlines()
   print(len(samples))
   model, tokenizer = get_model(False)
   print("1. Loading custom files")
   json_ds = load_dataset("json", data_files=f"/mnt/osmanthus/aklharas/speechBSD/transformers/jsons/test.json")
   print("2. Creating custom uncached files")
   dataset = json_ds.map(preprocess_testset, fn_kwargs={"tokenizer": tokenizer, 'samples': samples})
   print(dataset['train'][0])
   print("3. Saving to disk")
   dataset.save_to_disk(f"/mnt/osmanthus/aklharas/speechBSD/transformers/mt6-cascade-testset-region/test_en.data")

def preprocess_testset(batch, tokenizer, samples):
   global index
   print(samples[index])
   temp = tokenizer(samples[index], text_target=batch['ja_sentence'], truncation=True, padding=True, max_length=128)
   tag = "<" + batch["en_spk_state"] + ">"
   batch['input_ids'] = temp['input_ids']
   batch['labels'] = temp['labels']
   print(batch)
   tag_id = tokenizer.convert_tokens_to_ids([tag])
   batch['input_ids'].insert(1, tag_id[0])
   index = index+1
   print(index)
   print(batch)
   #return {k: torch.tensor(v) for k,v in batch.items()}
   return batch



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
        dataset.save_to_disk(f"/mnt/osmanthus/aklharas/speechBSD/transformers/mt5-region/{set_name}_{LANG}.data")

source_lang = "en"
target_lang = "ja"
metric = evaluate.load("sacrebleu", tokenize="ja-mecab")
def preprocess_function(examples, tokenizer):
    inputs = examples[f"{source_lang}_sentence"]
    targets = examples[f"{target_lang}_sentence"]
    tag = "<" + examples[f"en_spk_gender"] + ">"
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ja_XX"
    models_inputs = tokenizer(inputs, text_target=targets, truncation=True, padding=True, max_length=128)
    #tag_id = tokenizer.convert_tokens_to_ids([tag])
    #models_inputs['input_ids'].insert(1, tag_id[0])
    #models_inputs['attention_mask'].append(1)
    return models_inputs


def run(train=False, test=False, eval=False):
    wandb.init()
    model, tokenizer = get_model(True)
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

    train_ds = load_from_disk(f"{PATH}/mt5-region/train_en.data/train")
    validation_ds = load_from_disk(f"{PATH}/mt5-region/validation_en.data/train")
#    validation_ds = validation_ds.select(range(20))
    #test_ds = load_from_disk(f"{PATH}/mt6-cascade-testset-region/test_en.data/train")
    test_ds = load_from_disk(f"{PATH}/mt6-50base/test_en.data/train")
   # test_ds = test_ds.select(range(20))

    def compute_metrics(pred):
     preds, labels = pred
     print(pred)

     # decode preds and labels
     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip().replace("\n", ""))) for pred in decoded_preds]
     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip().replace("\n", ""))) for label in decoded_labels]
     gold = [[l] for l in decoded_labels]
     bleu = BLEU(tokenize="ja-mecab")
     sacrebleu = evaluate.load("sacrebleu")
     bleu_score = sacrebleu.compute(predictions=decoded_preds, references=gold, tokenize='ja-mecab')
     result = metric.compute(predictions=decoded_preds, references=gold)
     sacrebleu_score = bleu.corpus_score(decoded_preds, gold)
     print(bleu_score["score"])
     for i in range(20):
       print(f"{decoded_preds[i]} ---- {decoded_labels[i]}")
     print(bleu.get_signature())
     wandb.log({"bleu": sacrebleu_score.score})
     return result

    train_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/osmanthus/aklharas/checkpoints/mt5-region",
        evaluation_strategy="steps",
        predict_with_generate=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
        eval_accumulation_steps=2,
        gradient_accumulation_steps=8,
        num_train_epochs=30,
        bf16=True,
        save_steps=700,
        eval_steps=700,
        logging_steps=700,
        learning_rate=4e-5,
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
        #tokenizer.save_pretrained('/mnt/osmanthus/aklharas/models/mt4-50base')
        #trainer.save_model('/mnt/osmanthus/aklharas/models/mt4-50base')
    elif eval:
       trainer.evaluate()
    elif test:
       trainer.predict(test_ds)

def cascade_inference():
    model, tok = get_model(True)
    model.to(torch.device("cuda"))
    with open("asr.txt", "r") as f:
        samples = f.readlines()
    l = []
    for i, text in enumerate(samples):
     print(f"tranlsating {i}")
     tokenized = tok(text, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True).to(torch.device("cuda"))
     res = model.generate(**tokenized, decoder_start_token_id=tok.lang_code_to_id["ja_XX"])
     translated = tok.batch_decode(res, skip_special_tokens=True)
     l.append(translated)
    with open('hyp.txt', "w") as f1:
        f1.write("\n".join(l))



if __name__ == "__main__":
    #generate_cascade_testset()
    #generate_datasets()
    run(test=True)
    #cascade_inference()

