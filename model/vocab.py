import rdkit
import rdkit.Chem as Chem
import copy
import torch
from model.chemtools import *

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = [x for x in smiles_list] #copy
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)

class DVocab(object):

    def __init__(self, smiles_list, bond_list, cuda=True, trainflag=True):
        self.fragbond = [x for x in smiles_list]
        #frag = list(zip(*smiles_list))[0]
        frag = [x[0] for x in smiles_list]
        self.vocab = sorted(list(set(frag)))
        self.vmap = {x:i for i,x in enumerate(self.vocab)}

        self.fragment = [(x[0],x[1]) for x in smiles_list] #copy
        self.ivocab = sorted(list(set(self.fragment)))
        self.imap = {x:i for i,x in enumerate(self.ivocab)}
        self.iimap = {x[1]:i for i,x in enumerate(self.ivocab)}
        
        self.bondall = [x for x in bond_list] #copy
        self.bond = [(x[0]) for x in bond_list] #copy
        self.bond = sorted(list(set(self.bond)))
        self.bmap = {x:i for i,x in enumerate(self.bond)}
        self.cuda = cuda

        self.bondmaskwithcls = {}
        for cls in self.vocab :
            mask = torch.zeros(1, len(self.bond))
            self.bondmaskwithcls[self.vmap[cls]] = mask

        for cls in self.fragbond :
            bond = cls[2]
            bond = bond.split('-')
            bond = bond[1]+'-'+bond[0]        
            self.bondmaskwithcls[self.vmap[cls[0]]][0][self.bmap[bond]] = 10000.0

        self.vocabmask = {}
        for bond in self.bond :
            mask = torch.zeros(1, len(self.vocab))
            for y in self.fragbond :
                if y[2] == bond :
                    mask[0][self.vmap[y[0]]] = 1.0
            self.vocabmask[bond] = mask
        
        self.ivocabmask = {}
        for bond in self.bond :
            mask = torch.zeros(1, len(self.ivocab))
            for y in self.fragbond :
                if y[2] == bond :
                    mask[0][self.imap[(y[0],y[1])]] = 1.0
            self.ivocabmask[bond] = mask

        if trainflag :
            self.mask = {}
            for k, kk in enumerate(self.vocab) :
                mask = torch.zeros(1, len(self.ivocab))
                self.mask[k] = mask
            for g, gg in enumerate(self.ivocab) :
                self.mask[self.vmap[gg[0]]][0][g] = 10000.0

#        self.mask = {}
#        for k, kk in enumerate(self.vocab) :
#            mask = torch.zeros(1, len(self.ivocab))
#            for g, gg in enumerate(self.ivocab) :
#                if gg[0] == kk :
#                    mask[0][g] = 1000.0
#            mask = mask - 1000.0
#            self.mask[k] = mask
                  
#        self.mask = torch.zeros(len(self.vocab), len(self.ivocab))
#        for h,s in self.ivocab:
#            hid = self.vmap[h]
#            idx = self.imap[(h,s)]
#            self.mask[hid, idx] = 1000.0

#        if cuda: self.mask = self.mask.cuda()   
#        self.mask = self.mask - 1000.0
      
    def __getitem__(self, x):
        if (x[0],x[2]) in self.fragment :
            return self.vmap[x[0]], self.imap[(x[0],x[2])]
        elif (x[1],x[2]) in self.fragment :
            return self.vmap[x[1]], self.imap[(x[1],x[2])]
        elif x[0] in self.vocab and (x[0],x[2]) not in self.fragment :
            return self.vmap[x[0]], -1
        else :
            print('can not get fragment id!',x)
            
    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_ismiles(self, idx):
        return self.ivocab[idx][1]

    def get_ismilesid(self, ismiles) :
        return self.iimap[ismiles]
                
    def get_bond(self, idx):
        return self.bond[idx]

    def get_bondid(self, bond):
        return self.bmap[bond]

    def get_bondwithatom(self, atom):
        bond = []
        for x in self.bond :
            bii = x.split('-')
            if atom == bii[0] :
                bond.append(x)
        return bond

    def get_frag_bondid(self, x):
        bondlist = []
        for mo in self.fragbond :
            if x[0] in mo and x[1] in mo:
                bondlist.append(self.bmap[mo[2]])
        return bondlist

#    def get_frag_bonds(self, x):
#        bondlist = []
#        for mo in self.fragbond :
#            if x in mo:
#                bondlist.append(mo[2])
#        return list(set(bondlist))

    def get_frag_bonds(self, x):
        bondlist = []
        for mo in self.bondall :
            if x in mo:
                bondlist.append(mo[0])
        return list(set(bondlist))
        
    def get_ifrag_bonds(self, x, y):
        bondlist = []
        for mo in self.fragbond :
            if x in mo and y in mo:
                bondlist.append(mo[2])
        return bondlist
       
    def size(self):
        return len(self.vocab), len(self.ivocab)

    def bondsize(self):
        return len(self.bond)

    def get_bondwithdummy(self, atomlist, ismiles) :
        mol = Chem.MolFromSmiles(ismiles)
        for atom in mol.GetAtoms() :
            if atom.GetAtomMapNum() == 1 :
                bondatom = atom.GetSymbol()
        bond = atomlist[0]+'-'+bondatom
        if bond in self.bond : 
            return self.bmap[bond], bond
        else :
            return 0, 0

    def get_mask(self, cls_idx):  
        icmask = torch.zeros(len(cls_idx), len(self.ivocab))
        for x in range(len(cls_idx)) :
            icmask[x] = icmask[x] + self.mask[cls_idx[x].cpu().item()]
        if self.cuda: icmask = icmask.cuda()   
        icmask = icmask - 10000.0
        return icmask

#    def get_assmbond_mask(self, preassm) :
#        assmmask = torch.zeros(len(preassm), len(self.bond))
#        if len(preassm) == 0 :
#            assmmask = assmmask + 10000.0 
#        else :
#            for i, assm in enumerate(preassm) :
#                for x in assm :
#                    assmmask[i][self.bmap[x]] = 10000.0
        
#        if self.cuda: assmmask = assmmask.cuda()   
#        assmmask = assmmask - 10000.0
        
#        return assmmask

    def get_assmbond_mask(self, cls_labs) :
        assmmask = torch.zeros(len(cls_labs), len(self.bond))
        if len(cls_labs) == 0 :
            assmmask = assmmask + 10000.0 
        else :
            for i in range(len(cls_labs)) :
                assmmask[i] = assmmask[i] + self.bondmaskwithcls[cls_labs[i].cpu().item()]
        
        if self.cuda: assmmask = assmmask.cuda()   
        assmmask = assmmask - 10000.0
        
        return assmmask

    def get_curbond_mask(self, prebond) :
        assmmask = torch.zeros(1, len(self.bond))
        for x in prebond :
            assmmask[0][self.bmap[x]] = 10000.0

        if self.cuda: assmmask = assmmask.cuda()   
        assmmask = assmmask - 10000.0        
        return assmmask
    	   	
#    def get_cls_mask(self, preassm) :
#        clsmask = torch.zeros(len(preassm), len(self.vocab))
#        if len(preassm) > 0 : 
#            for i, assm in enumerate(preassm) :
#                if len(assm) > 0 :
#                    for x in assm :
#                        bond = x.split('-')
#                        bond = bond[1]+'-'+bond[0]
#                        clsmask[i] = clsmask[i] + self.vocabmask[bond]
#                    clsmask[i]= torch.where(clsmask[i] == 0,  torch.full_like(clsmask[i], -10000.0), clsmask[i])
#                    clsmask[i]= torch.where(clsmask[i] > 0,  torch.full_like(clsmask[i], 0), clsmask[i])
        
#        if self.cuda: clsmask = clsmask.cuda()           
#        return clsmask                

#    def get_icls_mask(self, preassm, cls_idx) :
#        iclsmask = torch.zeros(len(preassm), len(self.ivocab))
#        if len(preassm) > 0 :
#            for i, assm in enumerate(preassm) :
#                if len(assm) > 0 :
#                    for x in assm :
#                        bond = x.split('-')
#                        bond = bond[1]+'-'+bond[0]
#                        iclsmask[i] = iclsmask[i] + self.ivocabmask[bond]  
#                    iclsmask[i] = iclsmask[i] + self.mask[cls_idx[i].cpu().item()] - 10000.0
#                    iclsmask[i]= torch.where(iclsmask[i] <= 0,  torch.full_like(iclsmask[i], -10000.0), iclsmask[i])
#                    iclsmask[i]= torch.where(iclsmask[i] > 0,  torch.full_like(iclsmask[i], 0), iclsmask[i])
                                                
#        if self.cuda: iclsmask = iclsmask.cuda()     
#        return iclsmask   

    def get_cls_mask(self, preassm) :
        clsmask = torch.zeros(1, len(self.vocab))
        for bb in preassm :
            bond = self.get_bond(bb)
            bond = bond.split('-')
            bond = bond[1]+'-'+bond[0]
            clsmask = clsmask + self.vocabmask[bond]
        clsmask= torch.where(clsmask == 0,  torch.full_like(clsmask, -10000.0), clsmask)
        clsmask= torch.where(clsmask > 0,  torch.full_like(clsmask, 0), clsmask)
        
        if self.cuda: clsmask = clsmask.cuda()           
        return clsmask                

    def get_icls_mask(self, preassm, cls_idx) :
        iclsmask = torch.zeros(1, len(self.ivocab))
        for bb in preassm :
            bond = self.get_bond(bb)
            bond = bond.split('-')
            bond = bond[1]+'-'+bond[0]
            iclsmask = iclsmask + self.ivocabmask[bond]  
        iclsmask = iclsmask + self.mask[cls_idx.cpu().item()] - 10000.0
        iclsmask= torch.where(iclsmask <= 0,  torch.full_like(iclsmask, -10000.0), iclsmask)
        iclsmask= torch.where(iclsmask > 0,  torch.full_like(iclsmask, 0), iclsmask)
                                                
        if self.cuda: iclsmask = iclsmask.cuda()     
        return iclsmask   

        
COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1), ('*', 0)]
common_atom_vocab = Vocab(COMMON_ATOMS)

def count_inters(s):
    mol = Chem.MolFromSmiles(s)
    inters = [a for a in mol.GetAtoms() if a.GetAtomMapNum() > 0]
    return max(1, len(inters))


