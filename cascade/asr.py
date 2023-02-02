from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2FeatureExtractor
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datasets import load_from_disk, load_metric, load_dataset
import torch
import numpy as np
import re
import wandb
import torchaudio
from evaluate import load
import os
os.environ["WANDB_PROJECT"] = "ASR"

LANG = "en"
MODEL_ID = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
PATH = "/mnt/osmanthus/aklharas/speechBSD/transformers"
LOCAL = "/mnt/osmanthus/aklharas/checkpoints/asr/checkpoint-25000"


def get_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(LOCAL, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id, )
    return model, processor


def run(train=False, eval=False, test=False):
    wandb.init()
    model, processor = get_model()
    print(next(model.parameters()).device)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if cuda else torch.device('cpu')
    model.to(device)
    print(f"CUDA available: {cuda}, device to {device}")
    print(next(model.parameters()).device)
    model.freeze_feature_encoder()

    train_ds = load_from_disk(f"{PATH}/asr/train_en.data/train")
    validation_ds = load_from_disk(f"{PATH}/asr/validation_en.data/train")
    test_ds = load_from_disk(f"{PATH}/asr/test_en.data/train")
    train_ds = train_ds.remove_columns(['ja_sentence', 'ja_spkid', 'en_spkid', 'en_speaker', 'ja_speaker', 'no', 'ja_wav', 'en_wav'])
    validation_ds = validation_ds.remove_columns(['ja_sentence', 'ja_spkid', 'en_spkid', 'en_speaker', 'ja_speaker', 'no', 'ja_wav', 'en_wav'])
    test_ds = test_ds.remove_columns(['ja_sentence', 'ja_spkid', 'en_spkid', 'en_speaker', 'ja_speaker', 'no', 'ja_wav', 'en_wav'])

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        print(pred_ids)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        with open('asr.txt', "w") as f:
           f.write("\n".join(pred_str))

        wer = load("wer")
        score = wer.compute(predictions=pred_str, references=label_str)
        print("PRED VS GOLD")
        for i in range(20):
         print(f"{pred_str[i]} ---- {label_str[i]}")
        wandb.log({"wer": score})
        return {"wer": score}

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """

        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    train_args = TrainingArguments(
        output_dir="/mnt/osmanthus/aklharas/checkpoints/asr",
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        #group_by_length=True,
        num_train_epochs=10,
        fp16=True,
        #gradient_checkpointing=True,
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

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=train_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        tokenizer=processor.feature_extractor,
    )

    if train:
        #trainer.train()
        trainer.train(resume_from_checkpoint="/mnt/osmanthus/aklharas/checkpoints/asr/checkpoint-6000")
    elif eval:
        trainer.evaluate()
    elif test:
        trainer.predict(test_ds) 
#        with torch.no_grad():
#            test_ds = test_ds.select(range(100))
#            res = trainer.predict(test_ds)
#            print(res)


def generate_datasets():
    model, processor = get_model()
    sets = ['train', 'test', 'validation']

    for set_name in sets:
        print("1. Loading custom files")
        json_ds = load_dataset("json", data_files=f"/mnt/osmanthus/aklharas/speechBSD/transformers/{set_name}.json")
        print("2. Creating custom uncached files")
        dataset = json_ds.map(normalize_inputs, fn_kwargs={"split_type": f"{set_name}", "processor": processor})
        dataset.remove_columns(['ja_sentence', 'en_speaker', 'ja_speaker', 'no', 'ja_spkid', 'en_spkid', 'ja_wav', 'en_wav'])
        print("3. Saving to disk")
        dataset.save_to_disk(f"/mnt/osmanthus/aklharas/speechBSD/transformers/{set_name}_{LANG}.data")


def normalize_inputs(batch, split_type, processor):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    batch[f"{LANG}_sentence"] = re.sub(chars_to_ignore_regex, '', batch[f"{LANG}_sentence"]).lower()

    filename = batch[f"{LANG}_wav"]
    speech, sampling_rate = torchaudio.load(f"/mnt/osmanthus/aklharas/speechBSD/wav/{split_type}/{filename}")
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
    input_values = processor(resampler(speech).squeeze().numpy(), sampling_rate=16_000).input_values[0]
    batch["input_values"] = input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch[f"{LANG}_sentence"]).input_ids

    return batch


if __name__ == "__main__":
    #generate_datasets()
    run(test=True)
