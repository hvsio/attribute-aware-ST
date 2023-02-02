import argparse
from datetime import date

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_model_config", type=str)
    parser.add_argument("--nlp_model_config", type=str)
    parser.add_argument("--cache", action='store_true')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--field", type=str)
    parser.add_argument("--train_split", type=str)
    parser.add_argument("--test_split", type=str)
    parser.add_argument("--notes", type=str)
    parser.add_argument("--grad_accum", default=3, type=int)
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--unfreeze_warmup_steps", default=1000, type=int)
    parser.add_argument("--save_total_limit", default=2, type=int)
    parser.add_argument("--max_grad_norm", default=10, type=int)
    parser.add_argument("--worker", default=10, type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eval_step", default=700, type=int)
    parser.add_argument('--share_layer_ratio', default=0, type=float)
    parser.add_argument('--down_scale', default=8, type=int)
    parser.add_argument('--weighted_sum', action='store_true')
    parser.add_argument('--fixed_parameters', default=True, action='store_true')
    parser.add_argument("--custom_set_path", type=str)
    parser.add_argument("--max_input_length_in_sec", default=20, type=int)
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--modelpath", default=date.today().strftime("%d-%m-%Y--%H-%M"), type=str)
    parser.add_argument("--local", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--encdec", type=str)
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--vanilla", default=False, type=bool)
    parser.add_argument("--source_lang", default="en", type=str)
    parser.add_argument('--fixed_except', nargs='+',
                        default=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                                 "layernorm_embedding", 'attention', 'encoder'])
    parser.add_argument("--fp16", action='store_true')

    input_args, model_arg = parser.parse_known_args(args)
    input_args = {k: v for k, v in vars(input_args).items() if v is not None}
    other_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_args, other_arg
