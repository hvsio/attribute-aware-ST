from transformers import Bart

def get_model():
    processor = Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=False)
    model = Wav2Vec2ForCTC.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english', ctc_loss_reduction="mean",
                                           pad_token_id=processor.tokenizer.pad_token_id, )
    return model, processor