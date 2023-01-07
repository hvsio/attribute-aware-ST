import random

import torch
from transformers import SpeechEncoderDecoderModel, Wav2Vec2Processor
from datasets import load_from_disk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import evaluate

sacrebleu = evaluate.load("sacrebleu")

model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-1b-en-to-15")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-1b-en-to-15")

ds = load_from_disk("../speechBSD/transformers/test_cpu_en_HF_EED_mbart.data/train")

rnd = int(random.uniform(0, 2000))
inputs = processor(ds[rnd]["input_values"], sampling_rate=16000, return_tensors="pt")
generated_ids = model.generate(inputs["input_values"], attention_mask=inputs["attention_mask"], forced_bos_token_id=250012)
transcription = processor.batch_decode(generated_ids)
transcription1 = transcription[0][10:-4]
gold = ds[rnd]['ja_sentence']
weights = [
    (1./2., 1./2.),
    (1./3., 1./3., 1./3.),
    (1./4., 1./4., 1./4., 1./4.)
]
smoothing = SmoothingFunction()
print(sentence_bleu(gold.split(), transcription1, weights, smoothing_function=smoothing.method4))
print(sacrebleu.compute(references=[gold], predictions=transcription))
