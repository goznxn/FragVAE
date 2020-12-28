import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

#from hgraph import *
from model import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--frag', required=True)
parser.add_argument('--bond', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model_folder', required=True)
parser.add_argument('--fragment_smiles', required=True)
parser.add_argument('--radius', required=True)
parser.add_argument('--mol_file', required=True)

parser.add_argument('--topp', type=float, default=0.95)
parser.add_argument('--num_decode', type=int, default=1)
parser.add_argument('--mol_num', type=int, default=1)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--fragment_num', type=int, default=5)

parser.add_argument('--sample', action='store_true')
parser.add_argument('--novi', action='store_true')
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--fragment_size', type=int, default=50)
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--latent_size', type=int, default=250)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=20)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()
args.enum_root = True
args.greedy = not args.sample

#args.test = [line.strip("\r\n ") for line in open(args.test)]
with open(args.frag) as f1:
    vocab = [x.strip("\r\n ").split() for x in f1]
with open(args.bond) as f2:
    bond = [x.strip("\r\n ").split() for x in f2]
    
args.frag = DVocab(vocab, bond) 

model = MultiVAE(args).cuda()

model.load_state_dict(torch.load(args.model_folder))
model.eval()

torch.manual_seed(args.seed)
random.seed(args.seed)

with torch.no_grad(): 
    f1=open(args.mol_file,'a')
    for _ in tqdm(range( args.num_decode // args.mol_num)):
        smiles_list = model.sample(args.fragment_smiles, args.radius, args.batch_size, args.beam_size, args.topp, args.mol_num, args.enum_root, args.fragment_num, args.greedy)
        for _,smiles in enumerate(smiles_list):
            f1.write(smiles+'\n')
    f1.close()