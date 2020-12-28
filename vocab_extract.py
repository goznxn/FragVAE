import sys
from model import *
from rdkit import Chem
from multiprocessing import Pool
import argparse
from functools import partial

def process(data,radius):
    vocab = set()
    vocabbond = set()
    bondlist = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = HierGraph(s,radius)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['label']
            vocab.add( attr['label'] )
            for kk in attr['vocab'] : 
                vocabbond.add((kk))
            bond = attr['inter_bond']
            for x,y in bond :
                
                bondlist.add((x,y))
            
    return vocab, vocabbond, bondlist 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--moldata', required=True)
    parser.add_argument('--frag_file', required=True)
    parser.add_argument('--bond_file', required=True)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--ncpu', type=int, default=32)
    args = parser.parse_args()
    
    data = []
    for line in open(args.moldata) :
        data.append(line.strip('\n').strip().split()[1])
    data = list(set(data))

    ncpu = args.ncpu
    batch_size = len(data) // ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(ncpu)
    func = partial(process, radius = args.radius)
    datavocab = pool.map(func, batches)

    vocab_list, vocab_all, bond_list = [],[], []
    for i in range(ncpu) :
        vocab_list.extend(datavocab[i][0])
        vocab_all.extend(datavocab[i][1])
        bond_list.extend(datavocab[i][2])
    
    vocab_frag, vocab_bond = [], []
    for x in vocab_all :
        vocab_frag.append((x[0],x[2],x[3],x[4]))
        vocab_bond.append((x[3],x[4],x[1]))

    vocab_frag = list(set(vocab_frag))
    vocab_bond = list(set(vocab_bond))

    f1=open(args.frag_file,'a')
    for w, x, y, z in sorted(vocab_frag):
        f1.write(w+' '+x+' '+y+'-'+z+'\n')
    f1.close()

    f1=open(args.bond_file,'a')
    for w, x, y in sorted(vocab_bond):
        f1.write(w+'-'+x+' '+y+'\n')
    f1.close()