from transformers import MBart50Tokenizer, MBartForConditionalGeneration

m = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="ja_XX")
mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

print(m)

senence = "I love coffee"
label = "Kocham kawe"
tokenized = m(senence, return_tensors="pt", add_special_tokens=True)
r = m.prepare_seq2seq_batch(src_texts=senence, src_lang="en_XX", tgt_lang="ja_XX", tgt_texts=label, max_length=32, max_target_length=32)
#print(r)
#print(tokenized)
#print(m.convert_ids_to_tokens(tokenized))
res = mbart.generate(**tokenized, forced_bos_token_id=m.lang_code_to_id["ja_XX"])
#print(m.batch_decode(res))
#print(res)


#m.tgt_lang = "pl_PL"
tokenized = m(senence, return_tensors="pt", add_special_tokens=True)

#print(tokenized)
#print(m.convert_ids_to_tokens(tokenized))
res = mbart.generate(**tokenized, forced_bos_token_id=m.lang_code_to_id["pl_PL"])
print(m.batch_decode(res))