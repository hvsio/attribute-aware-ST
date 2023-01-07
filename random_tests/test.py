from hf_model import HFSpeechMixEEDmBart, SpeechMixConfig
import torch
import datasets
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test_cpu_en_HF_EED_mbart.data")
    parser.add_argument("--checkpoint", type=str, default="2800")
    parser = parser.parse_args()
    return parser



def main(args=None):
    known_args = parse_args(args)

    ds = datasets.load_from_disk(f'speechBSD/transformers/{known_args.dataset}/train')
    ds = ds.select(range(10))

    config = SpeechMixConfig.from_json_file(f'./checkpoints/checkpoint-{known_args.checkpoint}/config.json')
    checkpoint = torch.load(f'./checkpoints/checkpoint-{known_args.checkpoint}/pytorch_model.bin', map_location=torch.device('cpu'))
    model = HFSpeechMixEEDmBart(config)
    model.load_state_dict(checkpoint, strict=False)

    example = torch.FloatTensor(ds['input_values'][0])[None, :]
    label = torch.tensor(ds['labels'][0])[None, :]
    res = model(example, labels=label)

    pred_ids = [i[i != -100] for i in res.logits]
    pred_str = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
    print(pred_str)

    label_ids = [i[i != -100] for i in label]
    label_str = model.tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
    print(label_str)

if __name__ == "__main__":
    main()

