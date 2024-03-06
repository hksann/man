import argparse
import numpy as np
import torch

from models.models import BaseCMNModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')
    parser.add_argument('--diversity_lambda', type=float, default=0.5, help='diversity penalty for beam search')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='AdamW', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the encoder-decoder.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.999), help='betas for Adam optimizer.')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='eps for Adam optimizer.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='Whether to use the AMSGrad variant of Adam.')

    # Noam Optimizer Specific Args
#     parser.add_argument('--noamopt_warmup', type=int, default=5000, help='Number of warmup steps for NoamOpt.')
#     parser.add_argument('--noamopt_factor', type=int, default=1, help='Factor for NoamOpt.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size for StepLR.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma for StepLR.')

    # ReduceLROnPlateau Specific Args
    parser.add_argument('--reduce_factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced. new_lr = lr * factor.')
    parser.add_argument('--reduce_patience', type=int, default=10, help='Number of epochs with no improvement after which learning rate will be reduced.')
    parser.add_argument('--reduce_verbose', type=bool, default=True, help='If True, prints a message to stdout for each update.')
    parser.add_argument('--reduce_lr_threshold', type=float, default=0.01, help='Threshold for measuring the new optimum, to only focus on significant changes.')
    parser.add_argument('--reduce_cooldown', type=int, default=2, help='Number of epochs to wait before resuming normal operation after lr has been reduced.')
    parser.add_argument('--reduce_min_lr', type=float, default=1e-7, help='A lower bound on the learning rate of all param groups.')
    parser.add_argument('--reduce_eps', type=float, default=1e-8, help='Minimal decay applied to lr.')
    parser.add_argument('--threshold_mode', type=str, default='rel', choices=['rel', 'abs'], help="Mode for the threshold in ReduceLROnPlateau: 'rel' for relative change, 'abs' for absolute change.")
    
    # warm-up
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs for GradualWarmupScheduler.')
    parser.add_argument('--multiplier', type=float, default=10, help='Multiplier for learning rate at the end of warmup.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Whether to use pin_memory')
    parser.add_argument('--alpha', type=float, default=0.7, help='balance factor for cosine similarity and dot product in attention.')

    args = parser.parse_args()
    # 现在可以安全地访问 args 中的值
    print("Learning rate scheduler:", args.lr_scheduler)
    print("optim:", args.optim)
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, pin_memory=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False, pin_memory=True)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False, pin_memory=True)

    # build model architecture
    model = BaseCMNModel(args, tokenizer, alpha=args.alpha)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # Pass the relevant parameters to Trainer
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler,
                      train_dataloader, val_dataloader, test_dataloader, args.lr_ve, args.lr_ed, args.step_size, args.gamma)
    trainer.train()

if __name__ == '__main__':
    main()
