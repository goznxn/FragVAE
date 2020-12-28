from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy

from model import HierGraph, common_atom_vocab, DVocab, HierGraphwithFragment
import rdkit

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize_pair(mol_batch, vocab,radius):
    x, y = zip(*mol_batch)
    x = HierGraphwithFragment.tensorize(x, vocab, common_atom_vocab, radius)
    y = HierGraph.tensorize(y, vocab, common_atom_vocab, radius)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--moldata', required=True)
    parser.add_argument('--frag', required=True)
    parser.add_argument('--bond', required=True)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    with open(args.frag) as f1:
        vocab = [x.strip("\r\n ").split() for x in f1]
    with open(args.bond) as f2:
        bond = [x.strip("\r\n ").split() for x in f2]
    
    args.frag = DVocab(vocab, bond, cuda=False, trainflag=False)

    pool = Pool(args.ncpu) 
    random.seed(1)

    with open(args.moldata) as f:
        data = [line.strip("\r\n ").split()[:2] for line in f]

    random.shuffle(data)

    batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    func = partial(tensorize_pair, vocab = args.frag, radius = args.radius)
    all_data = pool.map(func, batches)
    num_splits = max(len(all_data) // 1000, 1)

    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open(args.data_folder+'/'+'tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
