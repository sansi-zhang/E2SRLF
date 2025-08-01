import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from test_sub import Test

from valid import Valid
from utils import *
from einops import rearrange
import argparse

# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='../log/E2SRLF')
    parser.add_argument('--testset_dir', type=str, default='../dataset/')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--patchsize', type=int, default=192)
    parser.add_argument('--minibatch_test', type=int, default=32)
    parser.add_argument('--model_file', type=str, default='../param/')
    parser.add_argument('--model_path', type=str, default='/datahdd/zcl/3dgs/HDSuperLF_final/param/Net_at_x4_sub3/Net_at_x4_sub3_lr0.0015_n_steps4000_15000best_mse.pth.tar')
    parser.add_argument('--save_path', type=str, default='../Results/')  
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--trainset_dir', type=str, default='../dataset/training/')
    parser.add_argument('--validset_dir', type=str, default='../dataset/validation/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1.5e-3, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=4000, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--net', type=str, default='E2SRLF_SRL1', help='net mode selecting')
    parser.add_argument('--mode', type=str, default='train', help='train or test mode selecting')
    parser.add_argument('--best', type=str, default='mse', help='param select')
    parser.add_argument('--scale_factor', type=float, default=0.5, help='scale_factor')
    parser.add_argument('--load_continuetrain', type=bool, default=False, help='continue train')
    parser.add_argument('--test_data_mode', type=str, default='HCI', help='real_data, HCI, our_data, ours_new, new_data')
    
    return parser.parse_args()


def launch(cfg):
    if cfg.mode == 'train':
        Train.train(cfg)
    elif cfg.mode == 'test':
        Test.test(cfg)
    elif cfg.mode == 'whole':
        Train.train(cfg)
        Test.test(cfg)
    elif cfg.mode == 'valid':
        Valid.valid(cfg)


def main(cfg):
    launch(cfg)

if __name__ == "__main__":
    cfg = parse_args()
    if 'x1' in cfg.net:
        from train_x1 import Train
    else:
        if 'SRL1' in cfg.net: 
            from train_SRL1 import Train
        else:
            from train import Train
    main(cfg)

        

