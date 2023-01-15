from transformers import MBart50Tokenizer, MBartForConditionalGeneration, MBartTokenizer
import torch

m = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
m1 = MBart50Tokenizer.from_pretrained("facebook/mbart-large-cc25")
mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
# newtoks = {'additional_special_tokens': ['<F>', '<M>']}
# n = m.add_special_tokens(newtoks)
# mbart.resize_token_embeddings(len(m))
print(mbart.config.decoder_start_token_id)
# print(mbart.vocab_size)
# print(len(m))
# print(m.convert_tokens_to_ids('<F>'))
# print(m.convert_tokens_to_ids('<M>'))
#print(m1.get_vocab())

senence = "I love coffee, i dont know"
tgt = "Kocham kawe, nie wiem"
m.tgt_lang = "ja_XX"

for x in range(1):
    tokenized = m(senence, return_tensors="pt", add_special_tokens=True)
    r = m.prepare_seq2seq_batch(src_texts=senence, text_target=tgt, src_lang="en_XX", max_length=32, max_target_length=32)
    res = mbart.generate(**tokenized, decoder_start_token_id=m.lang_code_to_id["ja_XX"])
    print(r)
    print(res)
    print(m.batch_decode(res, skip_special_tokens=True))
print(m.convert_ids_to_tokens([30]))
print(m.convert_ids_to_tokens([2]))

# m.tgt_lang = "pl_PL"
# tokenized = m(senence, text_target="Kocham kawe" ,return_tensors="pt", add_special_tokens=True)
# tokenized1 = m(senence ,return_tensors="pt", add_special_tokens=True)
# tokenized2 = m("Kocham kawe" ,return_tensors="pt", add_special_tokens=True)
# print(tokenized)
# print(tokenized1)
# print(tokenized2)
# print(m.convert_ids_to_tokens(tokenized.input_ids[0]))
# print(m.convert_ids_to_tokens(tokenized.labels[0]))
# print(m.convert_ids_to_tokens(tokenized2.input_ids[0]))
print("----------")

#print(tokenized)
#print(m.convert_ids_to_tokens(tokenized.input_ids[0]))
res = mbart.generate(**tokenized, forced_bos_token_id=m.lang_code_to_id["pl_PL"])
print(m.batch_decode(res, skip_special_tokens=True))