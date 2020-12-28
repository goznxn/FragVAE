import torch
import rdkit.Chem as Chem
import networkx as nx
from model.hier_graph import HierGraph
from model.chemtools import *
from collections import defaultdict

class IncBase(object):

    def __init__(self, batch_size, node_fdim, edge_fdim, max_nodes=50, max_edges=100, max_nb=120):
        self.max_nb = max_nb
        self.graph = nx.DiGraph()
        self.graph.add_node(0) #make sure node is 1 index
        self.edge_dict = {None : 0} #make sure edge is 1 index

        self.fnode = torch.zeros(max_nodes * batch_size, node_fdim).long().cuda()
        self.fmess = self.fnode.new_zeros(max_edges * batch_size, edge_fdim)
        self.agraph = self.fnode.new_zeros(max_edges * batch_size, max_nb)
        self.bgraph = self.fnode.new_zeros(max_edges * batch_size, max_nb)

    def add_node(self, feature=None):
        idx = len(self.graph)
        self.graph.add_node(idx)
        if feature is not None:
            self.fnode[idx, :len(feature)] = feature
        return idx

    def set_node_feature(self, idx, feature):
        self.fnode[idx, :len(feature)] = feature

    def can_expand(self, idx):
        return self.graph.in_degree(idx) < self.max_nb

    def add_edge(self, i, j, feature=None):
        if (i,j) in self.edge_dict: 
            return self.edge_dict[(i,j)]

        self.graph.add_edge(i, j)
        self.edge_dict[(i,j)] = idx = len(self.edge_dict)

        self.agraph[j, self.graph.in_degree(j) - 1] = idx
        if feature is not None:
            self.fmess[idx, :len(feature)] = feature
    
        in_edges = [self.edge_dict[(k,i)] for k in self.graph.predecessors(i) if k != j]
        self.bgraph[idx, :len(in_edges)] = self.fnode.new_tensor(in_edges)

        for k in self.graph.successors(j):
            if k == i: continue
            nei_idx = self.edge_dict[(j,k)]
            self.bgraph[nei_idx, self.graph.in_degree(j) - 2] = idx

        return idx

    def update_edge_feature(self, i, j, feature=None):
        reidx = self.edge_dict[(i,j)]
        if feature is not None:
            self.fmess[reidx, :len(feature)] = feature            


class IncTree(IncBase):

    def __init__(self, batch_size, node_fdim, edge_fdim, max_nodes=50, max_edges=100, max_nb=240, max_sub_nodes=80):
        super(IncTree, self).__init__(batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb)
        self.cgraph = self.fnode.new_zeros(max_nodes * batch_size, max_sub_nodes)

    def get_tensors(self):
        return self.fnode, self.fmess, self.agraph, self.bgraph, self.cgraph, None 

    def register_cgraph(self, i, smiles, nodes, edges, attached):
        nodeid = [x[0] for x in nodes]
        self.cgraph[i, :len(nodeid)] = self.fnode.new_tensor(nodeid)
        self.graph.nodes[i]['smiles'] = smiles
        self.graph.nodes[i]['cluster'] = nodes
        self.graph.nodes[i]['cluster_edges'] = edges
        self.graph.nodes[i]['attached'] = attached

    def update_attached(self, i, attached): 
            used = list(zip(*attached))[0]
            for x in range(len(self.graph.nodes[i]['attached'])) :
                if self.graph.nodes[i]['attached'][x][0] == used[0] :
                    self.graph.nodes[i]['attached'][x] = (self.graph.nodes[i]['attached'][x][0],self.graph.nodes[i]['attached'][x][1] - 1)

    def get_cluster(self, node_idx):
        cluster = self.graph.nodes[node_idx]['cluster']
        edges = self.graph.nodes[node_idx]['cluster_edges']
        used = self.graph.nodes[node_idx]['attached']
        return cluster, edges, used

    def get_smiles(self, node_idx):
        return self.graph.nodes[node_idx]['smiles']

    def get_cluster_nodes(self, node_list):
        return [ c[0] for node_idx in node_list for c in self.graph.nodes[node_idx]['cluster'] ]

    def get_cluster_edges(self, node_list):
        return [ e for node_idx in node_list for e in self.graph.nodes[node_idx]['cluster_edges'] ]


class MessGraph(IncBase):

    def __init__(self, avocab, batch_size, node_fdim, edge_fdim, max_nodes=200, max_edges=400, max_nb=100):
        super(IncGraph, self).__init__(batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb)
        self.avocab = avocab
        self.mol = Chem.RWMol()
        self.mol.AddAtom( Chem.Atom('C') ) 
        self.fnode = self.fnode.float()
        self.fmess = self.fmess.float()
        self.batch = defaultdict(list)
        self.fragnum = []
        self.fragcore = []
        self.molnum = {}
        for i in range(batch_size) :
            self.molnum[i] = 0

    def get_mol(self):
        mol_list = [None] * len(self.batch)
        for batch_idx, batch_atoms in self.batch.items():
            mol = get_sub_mol(self.mol, batch_atoms)
            mol = sanitize(mol, kekulize=False)
            if mol is None: 
                mol_list[batch_idx] = None
            else:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                mol_list[batch_idx] = Chem.MolToSmiles(mol)
        length = 0
        for x in range(len(self.molnum)) :
            length = length + self.molnum[x]
        
        return mol_list

    def get_tensors(self):
        return self.fnode, self.fmess, self.agraph, self.bgraph, None 

    def add_mol(self, batch_idx, smiles, inter_label, nth_child, smilescore, clab):
        self.molnum[batch_idx]=self.molnum[batch_idx] +1
        self.fragnum.append(Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(smiles))))
        self.fragcore.append(smilescore)
        self.fragnum = list(set(self.fragnum))
        self.fragcore = list(set(self.fragcore))
        emol = get_mol(smiles)

        atom_map = {}
        new_atoms, new_bonds, attached, dummy = [], [], [], []
        inter_attach = []
        if len(inter_label) > 0 :
            for x,y in inter_label :
                flag = 0
                bond = y.split('-')   
                for atom in emol.GetAtoms():
                    if atom.GetSymbol() == '*':
                        for b in atom.GetNeighbors() : 
                            if b.GetAtomMapNum() == 1:
                                inter_attach.append((x,b.GetIdx()))
                                flag = 1
                                break
                    if flag ==  1 : break

        for atom in emol.GetAtoms(): 
            if atom.GetSymbol() == '*':
                for b in atom.GetNeighbors() : 
                    dummy.append(b.GetIdx())
            else:
                new_atom = copy_atom(atom)
                new_atom.SetAtomMapNum( batch_idx ) 
                idx = self.mol.AddAtom( new_atom )
                assert idx == self.add_node( self.get_atom_feature(new_atom) ) #mol and nx graph must have the same indexing
                atom_map[atom.GetIdx()] = idx
                new_atoms.append((idx,atom.GetSymbol()))
                self.batch[batch_idx].append(idx)

        for bond in emol.GetBonds():
            if bond.GetBeginAtom().GetSymbol() == "*" or bond.GetEndAtom().GetSymbol() == "*" : continue
            a1 = atom_map[bond.GetBeginAtom().GetIdx()]
            a2 = atom_map[bond.GetEndAtom().GetIdx()]
            if a1 == a2: continue
            bond_type = bond.GetBondType()
            existing_bond = self.mol.GetBondBetweenAtoms(a1, a2)
            if existing_bond is None:
                self.mol.AddBond(a1, a2, bond_type)
                self.add_edge(a1, a2, self.get_mess_feature(bond.GetBeginAtom(), bond_type, nth_child if a2 in attached else 0) ) #only child to father node (in intersection) have non-zero nth_child
                self.add_edge(a2, a1, self.get_mess_feature(bond.GetEndAtom(), bond_type, nth_child if a1 in attached else 0) ) 

            new_bonds.extend( [ self.edge_dict[(a1,a2)], self.edge_dict[(a2,a1)] ] )
        
        if len(inter_attach) > 0 :
            for x,y in inter_attach :
                a1 = atom_map[y]
                a2 = x
                atom1 = self.mol.GetAtomWithIdx(a1)
                atom2 = self.mol.GetAtomWithIdx(a2)
                bond_type = Chem.rdchem.BondType.SINGLE
                if self.mol.GetBondBetweenAtoms(a1, a2) is None: 
                    self.mol.AddBond(a1, a2, bond_type)
                    self.add_edge(a1, a2, self.get_mess_feature(atom1, bond_type, nth_child) ) 
                    self.add_edge(a2, a1, self.get_mess_feature(atom2, bond_type, nth_child) )

        for x in atom_map :
            if x not in dummy :
                attached.append((atom_map[x],0))
            else :
                dnum = dummy.count(x)
                if len(inter_label) > 0 :
                    if x == inter_attach[0][1] :
                        attached.append((atom_map[x],dnum-1))
                    else :
                        attached.append((atom_map[x],dnum))
                else :
                    attached.append((atom_map[x],dnum))         
        if emol.GetNumAtoms() == 1:
            attached = []
        return new_atoms, new_bonds, attached

    def add_beam_mol(self, batch_idx, copy_idx, smiles, inter_label, nth_child, smilescore, clab):
        self.fragnum.append(Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(smiles))))
        self.fragcore.append(smilescore)
        self.fragnum = list(set(self.fragnum))
        self.fragcore = list(set(self.fragcore))
        emol = get_mol(smiles)

        atom_map = {}
        new_atoms, new_bonds, attached, dummy = [], [], [], []
        inter_attach = []
        if len(inter_label) > 0 :
            for x,y in inter_label :
                flag = 0
                bond = y.split('-')   
                for atom in emol.GetAtoms():
                    if atom.GetSymbol() == '*':
                        for b in atom.GetNeighbors() : 
                            if b.GetAtomMapNum() == 1:
                                inter_attach.append((x,b.GetIdx()))
                                flag = 1
                                break
                    if flag ==  1 : break
        if len(self.batch[batch_idx]) == 0 :
            self.molnum[batch_idx]=self.molnum[copy_idx]
            for atomid in self.batch[copy_idx] :
                new_atom = copy_atom(self.mol.GetAtomWithIdx(atomid))
                idx = self.mol.AddAtom( new_atom )
                assert idx == self.add_node( self.get_atom_feature(new_atom) ) #mol and nx graph must have the same indexing
                atom_map[atomid] = idx 
                self.batch[batch_idx].append(idx)
                if atomid == inter_attach[0][0] :
                    attachnode = inter_attach[0][1]
                    inter_attach = []
                    inter_attach.append((idx,attachnode))

            for atomid1 in self.batch[copy_idx] :
                for atomid2 in self.batch[copy_idx] :
                    if atomid1 == atomid2 : continue
                    existing_bond = self.mol.GetBondBetweenAtoms(atomid1, atomid2)
                    if existing_bond is not None:
                        a1 = atom_map[atomid1]
                        a2 = atom_map[atomid2]                        
                        bond_type = existing_bond.GetBondType()
                        check_bond = self.mol.GetBondBetweenAtoms(a1, a2)
                        if check_bond is None:
                            self.mol.AddBond(a1, a2, bond_type)
                            self.add_edge(a1, a2, self.get_mess_feature(self.mol.GetAtomWithIdx(a1), bond_type, nth_child if a2 in attached else 0) ) #only child to father node (in intersection) have non-zero nth_child
                            self.add_edge(a2, a1, self.get_mess_feature(self.mol.GetAtomWithIdx(a2), bond_type, nth_child if a1 in attached else 0) ) 
                    
        self.molnum[batch_idx]=self.molnum[batch_idx] +1
        for atom in emol.GetAtoms(): 
            if atom.GetSymbol() == '*':
                for b in atom.GetNeighbors() : 
                    dummy.append(b.GetIdx())
            else:
                new_atom = copy_atom(atom)
                new_atom.SetAtomMapNum( batch_idx ) 
                idx = self.mol.AddAtom( new_atom )
                assert idx == self.add_node( self.get_atom_feature(new_atom) ) 
                atom_map[atom.GetIdx()] = idx
                new_atoms.append((idx,atom.GetSymbol()))
                self.batch[batch_idx].append(idx)

        for bond in emol.GetBonds():
            if bond.GetBeginAtom().GetSymbol() == "*" or bond.GetEndAtom().GetSymbol() == "*" : continue
            a1 = atom_map[bond.GetBeginAtom().GetIdx()]
            a2 = atom_map[bond.GetEndAtom().GetIdx()]
            if a1 == a2: continue
            bond_type = bond.GetBondType()
            existing_bond = self.mol.GetBondBetweenAtoms(a1, a2)
            if existing_bond is None:
                self.mol.AddBond(a1, a2, bond_type)
                self.add_edge(a1, a2, self.get_mess_feature(bond.GetBeginAtom(), bond_type, nth_child if a2 in attached else 0) ) 
                self.add_edge(a2, a1, self.get_mess_feature(bond.GetEndAtom(), bond_type, nth_child if a1 in attached else 0) ) 

            new_bonds.extend( [ self.edge_dict[(a1,a2)], self.edge_dict[(a2,a1)] ] )

        if len(inter_attach) > 0 :
            for x,y in inter_attach :
                a1 = atom_map[y]
                a2 = x
                atom1 = self.mol.GetAtomWithIdx(a1)
                atom2 = self.mol.GetAtomWithIdx(a2)
                bond_type = Chem.rdchem.BondType.SINGLE
                if self.mol.GetBondBetweenAtoms(a1, a2) is None: 
                    self.mol.AddBond(a1, a2, bond_type)
                    self.add_edge(a1, a2, self.get_mess_feature(atom1, bond_type, nth_child) ) 
                    self.add_edge(a2, a1, self.get_mess_feature(atom2, bond_type, nth_child) )
                    
        for x in atom_map :
            if x not in dummy :
                attached.append((atom_map[x],0))
            else :
                dnum = dummy.count(x)
                if x == inter_attach[0][1] :
                    attached.append((atom_map[x],dnum-1))
                else :
                    attached.append((atom_map[x],dnum))                

        if emol.GetNumAtoms() == 1: 
            attached = []
        return new_atoms, new_bonds, attached, atom_map

    def get_atom_feature(self, atom):
        f = torch.zeros(self.avocab.size())
        symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
        f[ self.avocab[(symbol,charge)] ] = 1
        return f.cuda()

    def get_mess_feature(self, atom, bond_type, nth_child):
        f1 = torch.zeros(self.avocab.size())
        f2 = torch.zeros(len(HierGraph.BOND_LIST))
        f3 = torch.zeros(HierGraph.MAX_POS)
        symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
        f1[ self.avocab[(symbol,charge)] ] = 1
        f2[ HierGraph.BOND_LIST.index(bond_type) ] = 1
        f3[ nth_child ] = 1
        return torch.cat( [f1,f2,f3], dim=-1 ).cuda()

    def get_assm_cands(self, cluster, used, smiles):
        attach_points = []
        emol = get_mol(smiles)
        if emol.GetNumAtoms() == 1:
            attach_points.append(0)
        else:
            for atom in emol.GetAtoms() :
                if atom.GetSymbol() == '*':
                    for b in atom.GetNeighbors() :
                        if b.GetAtomMapNum() == 1 :
                            attach_points.append(b.GetIdx())

        anchor_smiles = [smiles]
        cands = [ [x[0]] for x in cluster if x[0] not in used ]

        return cands, anchor_smiles, attach_points


