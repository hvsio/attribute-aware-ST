from transformers import MBartForConditionalGeneration, Wav2Vec2ForCTC, MBart50Tokenizer, Wav2Vec2Tokenizer
from datasets import load_from_disk
import torch
from nltk.translate.bleu_score import corpus_bleu

PATH = "/mnt/osmanthus/aklharas"
ASR_PATH = f"{PATH}/checkpoints/asr/checkpoint-"
MT_PATH = f"{PATH}/checkpoints/mt/checkpoint-"
EVAL_DS = f"{PATH}/speechBSD/transformers/validation_cuda:0_en_mbart_toklabel_nolower.data/train"

def run():
    mbart = MBartForConditionalGeneration.from_pretrined(MT_PATH)
    wav2vec = Wav2Vec2ForCTC.from_pretrained(ASR_PATH)
    device = torch.device("cuda")
    mbart.to(device)
    wav2vec.to(device)

    bart_tok = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="ja_XX")
    wavTok = Wav2Vec2Tokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

    inputs = load_from_disk(EVAL_DS)
    golden_sent = inputs["ja_sentence"]
    references = [[g] for g in golden_sent]
    asr_labels = inputs["en_sentence"]


## ASR
    resampled_wavs = torch.tensor(inputs["input_values"], device="cuda").unsqueeze(0)
    asr_tokend_ids = wav2vec(resampled_wavs).logits
    pred_ids = torch.argmax(asr_tokend_ids, dim=-1)
    asr_output = wavTok.batch_decode(asr_tokend_ids)[0]
    print([asr_output[i] for i in range(20)])

## MT
    #bart_input = bart_tok(asr_output, return_tensors="pt")
    #generated_tokens = mbart.generate(**bart_input, forced_bos_token_id=bart_tok.lang_code_to_id["ja_XX"])
    #hypotheses = bart_tok.batch_encode(generated_tokens, skip_special_tokens=False)
    #comparisons = zip(hypotheses, golden_sent)
    #for i in range(20):
    #   print(f"{hypotheses[i]} ---- {golden_sent[i]}\n")

    #bleu_score = corpus_bleu(references, hypotheses)



