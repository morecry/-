import argparse
from train import train
import random
import numpy as np
import torch

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--action", default='train', type=str, help='Train or test model.')
    parser.add_argument("--bert_model", default='bert', type=str, help='The path to load pre-trained bert model.')
    parser.add_argument("--output_dir", default='output', type=str, help='The path to save output log.')
    parser.add_argument("--data_dir", default='data', type=str, help='The path to sava dataset.')
    parser.add_argument("--use_pretrain", default='bert', type=str, help='use bert or personal pre-trained model')

    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=100, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    ## Significant parameter
    parser.add_argument("--train_batch_size", default=2048, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2048, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=25, type=int, help="Log frequency")
    parser.add_argument("--eval_freq", default=625, type=int, help="Eval frequency")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=65536, help="random seed for initialization")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    args = parser.parse_args()
    print(args)
    set_seed(args)

    if args.action == 'train':
        print("=" * 100 + "\n")
        print("Start training the model..." + "\n")
        print("=" * 100 + "\n")
        # train(args, type='mask')
        # train(args, type='nxt')
        args.num_train_epochs = 20
        train(args,type='match')
    else:
        print("Unexpected args.action!")