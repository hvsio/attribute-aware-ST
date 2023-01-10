from transformers import MBartForConditionalGeneration, Wav2Vec2ForCTC, MBart50Tokenizer, Wav2Vec2CTCTokenizer
from datasets import load_from_disk
import torch
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

PATH = "/mnt/osmanthus/aklharas"
ASR_PATH = f"{PATH}/checkpoints/asr/checkpoint-25000"
MT_PATH = f"{PATH}/checkpoints/mt-saved/checkpoint-2500"
EVAL_DS = f"/mnt/osmanthus/aklharas/speechBSD/transformers/validation_cuda:0_en_mbarttoklabel_nolower.data/train"

def run():
    print("Establishing models...")
    mbart = MBartForConditionalGeneration.from_pretrained(MT_PATH)
    wav2vec = Wav2Vec2ForCTC.from_pretrained(ASR_PATH)
    device = torch.device("cuda")
    mbart.to(device)
    wav2vec.to(device)

    bart_tok = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="ja_XX")
    wavTok = Wav2Vec2CTCTokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
 
    print("Establishing the dataset...")
    inputs = load_from_disk(EVAL_DS)
    golden_sent = inputs["ja_sentence"]
    references = [[g] for g in golden_sent]
    asr_labels = inputs["en_sentence"]


## ASR
    print("Starting ASR...")
    pred = []
    for i in range(len(inputs["input_values"])):
     print(f"Predicting {i}")
    #resampled_wavs = [torch.tensor(t, device="cuda").unsqueeze(0) for t in inputs["input_values"]]
     with torch.no_grad():
      asr_tokend_ids = wav2vec(torch.tensor(inputs["input_values"][i], device="cuda").unsqueeze(0))
      logits = asr_tokend_ids.logits
     pred_ids = torch.argmax(logits, dim=-1)
     asr_output = wavTok.batch_decode(pred_ids)
     pred.append(asr_output)
     print(asr_output)
    with open("./asr.txt", "w") as f:
      f.write("\n".join(asr_output))
    exit()

## MT
    print("Translating")
    bart_input = bart_tok(asr_output, return_tensors="pt").to(device)
    generated_tokens = mbart.generate(**bart_input, forced_bos_token_id=bart_tok.lang_code_to_id["ja_XX"])
    print("Predicting translations...")
    hypotheses = bart_tok.batch_decode(generated_tokens, skip_special_tokens=False)
    comparisons = zip(hypotheses, golden_sent)
    for i in range(20):
       print(f"{hypotheses[i]} ---- {golden_sent[i]}\n")

    bleu_score = corpus_bleu(references, hypotheses)



run()
