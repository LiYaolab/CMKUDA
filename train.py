import argparse
import torch
import os

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from torch.cuda import init

# custom
from dassl.data.datasets import VisDA17
from dassl.data.datasets import OfficeHome
from dassl.data.datasets import miniDomainNet

import trainers.CMKUDA

import warnings
warnings.filterwarnings("ignore")


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables for DAPL.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"
    cfg.MODEL.INIT_WEIGHTS_CTX = None
    cfg.MODEL.INIT_WEIGHTS_PRO = None
    cfg.TRAINER.DAPL = CN()
    cfg.TRAINER.DAPL.N_DMX = 16  # number of DSC tokens
    cfg.TRAINER.DAPL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.DAPL.CSC = False  # class-specific context
    cfg.TRAINER.DAPL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.DAPL.T = 1.0
    cfg.TRAINER.DAPL.TAU = 0.5
    cfg.TRAINER.DAPL.U = 1.0
    # cfg.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
    # cfg.DATALOADER.TRAIN_U.SAMPLER = "RandomSampler"
    cfg.TRAINER.CMKUDA = CN()
    cfg.TRAINER.CMKUDA.N_CTX = 32  # number of context vectors #16
    cfg.TRAINER.CMKUDA.N_CLS = 2 # number of class vectors
    cfg.TRAINER.CMKUDA.CSC = False  # class-specific context
    cfg.TRAINER.CMKUDA.PREC = "amp"  # fp16, fp32, amp #fp16
    cfg.TRAINER.CMKUDA.TAU = 0.5
    cfg.TRAINER.CMKUDA.U = 2.0 #1.0
    cfg.TRAINER.CMKUDA.IND = 1.0
    cfg.TRAINER.CMKUDA.IM = 1.0
    cfg.TEST.FINAL_MODEL = "best_val"
    cfg.OPTIM_C = cfg.OPTIM.clone()
    cfg.TRAINER.CMKUDA.STRONG_TRANSFORMS = []

    # cfg.TRAINER.CMKUDA.V_GAMMA = 0.01
    # cfg.TRAINER.CMKUDA.T_GAMMA = 0.01

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    print(cfg)
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/root/dataset", help="path to dataset")
    parser.add_argument("--output-dir",
                        default='/root/CMKUDA/output/counterfactual',
                        type=str,
                        help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=-1,
                        help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains",
                        type=str,
                        nargs="+",
                        help="source domains for DA/DG")
    parser.add_argument("--target-domains",
                        type=str,
                        nargs="+",
                        help="target domains for DA/DG")
    parser.add_argument("--transforms",
                        type=str,
                        nargs="+",
                        help="data augmentation methods")
    parser.add_argument("--config-file",
                        default='/root/CMKUDA/configs/trainers/CMKUDA/CMKUDA.yaml',
                        type=str,
                        help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        default='/root/CMKUDA/configs/datasets/office_home.yaml',
        type=str,
        help="path to config file for dataset setup",
    ) #/root/CMKUDA/configs/datasets/visda17.yaml
    #/root/CMKUDA/configs/datasets/office_home.yaml
    parser.add_argument("--trainer",
                        type=str,
                        default="CMKUDA",
                        help="name of trainer")
    parser.add_argument("--backbone",
                        type=str,
                        help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument("--load-epoch",
                        type=int,
                        help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train",
                        action="store_true",
                        help="do not call trainer.train()")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    print(args.opts)
    main(args)
