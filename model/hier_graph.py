import torch
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from model.chemtools import *
from model.tools import *

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

SLICE_SMARTS = {
    "extendedRECAP" : [

        "[NX2]=!@[NX2]",
        "[C;!D1;$(C(=!@[O,S]))]!@[O,S;!D1;!$([O,S][*;D1])]",
        "[N;!D1;!$(N-C=[#8]);$(N-!@[*;!D1])]-!@[*;!D1]",
        "[#7;R;D3]-!@[*;!D1]",
        "[n]-!@[C]",
        "[c,n]-!@[c]",
        "[#7;D2,D3]-!@[S;$([S](=[O])=[O])]",
        "[#6;!D1;!$([#6](=!@[#8,#16]))]-!@[#8,#16;$([#8,#16]!@[#6;!D1]);!$([#8,#16][#6]=!@[#8,#16])]",
        "[C;!D1]=!@[C;!D1]",
        "[S;!D1]-!@[S;!D1]",
        "[#6;R]-!@[#6;R]",
        "[*;!P;!D1][OX2;$([OX2][PX4])]",
        "[*;!D1][#16$([#16X4]([*;!D1])(=[OX1])=[OX1])]",
        "[#6;R]-!@[#6$([#6;H2]-!@[#6;R])]",
        "[*;!D1;!#8][#7;$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        "[*;!D1][S;D3;$(S(=O)(=O)),$(S(=O)([OH]))]",
        "[*;!S;!D1][OX2;$([OX2][S;$([#16X4](=[OX1])(=[OX1])([OX2])([OX1]))])]",
        "[*;!S;!D1][S;$([#16X4](=[OX1])(=[OX1])([OX1]))]",
        "[*][SD4;$(S(=O)(=O)([ND1]))]",
        "[#6;R][C$([CX3H1](=[O]))]",
        "[N;+1;D4]!@[#6]",
        "[$(N(@-C(=O)))]!@-[#6]",
        "[#6;R]-!@[*;!#1]",
        "[O;$(O(P)[*;!P])][PX4;$(P([OX2])(=O)[OX2])]",

        "[#7;D2,D3]!@[C;$(C(!@=[#8,#16])!@[#7;D2,D3])]",
        "[#7;!D1]!@[C;$(C=!@[#8,#16]);!$(C([#7])[#7])]",
        "[#6;$([#6]-!@[#6;R])]-!@[*;!#1]"        
    ]    
}
SLICE_SMARTS = {name: [Chem.MolFromSmarts(sma) for sma in smarts] for name, smarts in SLICE_SMARTS.items()}


class FragSliceEnumerator:

    def __init__(self, slice_smarts):
        self.slice_smarts = slice_smarts

    def enumerate(self, mol):
        Chem.Kekulize(mol)
        new_mol = Chem.RWMol(mol)
        atomlabel = {}
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx()+1)
            atomlabel[atom.GetIdx()+1] = atom.GetSymbol()
            
        matchesfilter, bondsymbol=[] , []
        matches = self._get_matches(new_mol)
        matchesfilter = self._update_matches_s(new_mol,matches)        
        to_cut_bonds = list(sorted(new_mol.GetBondBetweenAtoms(aidx, oaidx).GetIdx()                                       for aidx, oaidx in matchesfilter))
        attachment_point_idxs = [(0, 0) for i in range(len(to_cut_bonds))]

        cut_mol=[]
        cls = []

        if len(matchesfilter) > 0 :
            cut_mol = Chem.FragmentOnBonds(new_mol, bondIndices=to_cut_bonds, dummyLabels=attachment_point_idxs)
            flag=len(mol.GetAtoms())+1
            for aidx, oaidx in matchesfilter :
                for bond in new_mol.GetBonds():
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    a1id = a1.GetIdx()
                    a2id = a2.GetIdx()
                    if (aidx == a1id and oaidx == a2id) or (aidx == a2id and oaidx == a1id):
                        new_idx1 = new_mol.AddAtom(copy_atom(a1))
                        new_mol.GetAtomWithIdx(new_idx1).SetAtomMapNum(flag)
                        new_mol.AddBond(new_idx1, a2.GetIdx(), bond.GetBondType())
                        
                        new_idx2 = new_mol.AddAtom(copy_atom(a2))
                        new_mol.GetAtomWithIdx(new_idx2).SetAtomMapNum(flag)
                        new_mol.AddBond(a1.GetIdx(), new_idx2, bond.GetBondType())
                        new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())  
                        flag=flag+1  
                        break        
        else :
            cut_mol = new_mol         
                    
        new_mol = new_mol.GetMol()
        new_smiles = Chem.MolToSmiles(new_mol)
            
        for fragment in new_smiles.split('.'):
            fmol = Chem.MolFromSmiles(fragment)
            indices = set([atom.GetAtomMapNum() for atom in fmol.GetAtoms()])
            for atom in fmol.GetAtoms():
                atom.SetAtomMapNum(0)
                fsmiles = Chem.MolToSmiles(fmol)
            cls.append((indices))

        return matchesfilter, cls, cut_mol, atomlabel

    def _get_matches(self, mol):
        matches = set()
        for smarts in self.slice_smarts:
            matches |= set(tuple(sorted(match)) for match in mol.GetSubstructMatches(smarts))
        return list(matches)
                
    def _update_matches_s(self, mol, matches):
        matchesfilter = []
        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        for aidx, oaidx in matches :
            if mol.GetBondBetweenAtoms(aidx, oaidx).GetBondType() == Chem.rdchem.BondType.SINGLE :
                a1 = mol.GetAtomWithIdx(aidx)
                a2 = mol.GetAtomWithIdx(oaidx)
                if len(a1.GetNeighbors()) > 1 and len(a2.GetNeighbors()) > 1 :
                    cluflag = 0
                    for clu in ssr :
                        if aidx in clu and oaidx in clu :
                            cluflag = 1
                    if cluflag == 0 : matchesfilter.append((aidx, oaidx))
        return list(set(matchesfilter))

class HierGraph(object):

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 20

    def __init__(self, smiles, radius = 2, flag=1):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.flag = flag
        self.radius = radius

        self.mol_graph = self.build_mol_graph()
        self.clusters, self.atom_cls, self.matches, self.fragmol, self.atomlabel = self.find_clusters()
        self.mol_tree = self.tree_decomp()        
        self.order = self.label_tree()

    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        return new_atom

    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: 
            return [(0,)], [[0]]

        clusters = []
        if self.flag == 1 :
            Fragenumerator = FragSliceEnumerator(SLICE_SMARTS['extendedRECAP'])
            matches, clusters, fragmol, atomlabel = Fragenumerator.enumerate(mol)
            matches = [(x+1,y+1) for x,y in matches]

        else :
            matches, cls, clusters, atomlabel  = [],[],[], []

            for atom in mol.GetAtoms():
                cls.append(atom.GetIdx())
            clusters.append((cls))
            
            Chem.Kekulize(mol)
            new_mol = Chem.RWMol(mol)
            for atom in new_mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            
            fragmol = new_mol
   
        if 1 not in clusters[0]: #root is not node[0]
            for i,cls in enumerate(clusters):
                if 1 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    break

        atom_cls = [[] for i in range(len(self.mol.GetAtoms())+len(matches)+1)]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom_cls[atom].append(i)

        return clusters, atom_cls, matches, fragmol, atomlabel

    def tree_decomp(self):
        clusters = self.clusters
        graph = nx.empty_graph( len(clusters) )
        for atom, nei_cls in enumerate(self.atom_cls):
            if len(nei_cls) <= 1: continue
            for i,c1 in enumerate(nei_cls):
                for c2 in nei_cls[i + 1:]:
                    inter = set(clusters[c1]) & set(clusters[c2])
                    graph.add_edge(c1, c2, weight = len(inter))

        n, m = len(graph.nodes), len(graph.edges)
        assert n - m <= 1 
        return graph if n - m == 1 else nx.maximum_spanning_tree(graph)

    def label_tree(self):
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa 
            sorted_child = sorted([ y for y in self.mol_tree[x] if y != fa ]) #better performance with fixed order
            for idx,y in enumerate(sorted_child):
                self.mol_tree[x][y]['label'] = 0 
                self.mol_tree[y][x]['label'] = idx + 1 #position encoding
                prev_sib[y] = sorted_child[:idx] 
                prev_sib[y] += [x, fa] if fa >= 0 else [x]
                order.append( (x,y,1) )
                dfs(order, pa, prev_sib, y, x)
                order.append( (y,x,0) )

        order, pa = [], {}
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))]
        dfs(order, pa, prev_sib, 0, -1)

        order.append( (0, None, 0) ) #last backtrack at root

        for i,cls in enumerate(self.clusters):
            if pa[i] >= 0 :
                for j,x in enumerate(self.matches) :
                    if x[1] in self.clusters[pa[i]] and x[0] in cls:
                        self.matches[j] = (x[1],x[0])
            else :
                for j, x in enumerate(self.matches) :
                    if x[0] in cls :
                        self.matches[j] = (x[1],x[0])
                                        
        fragall=Chem.MolToSmiles(self.fragmol,True)
        fraglist = [Chem.MolFromSmiles(x) for x in fragall.split('.')]


        tree = self.mol_tree
        for i,cls in enumerate(self.clusters):
            clss = [atom for atom in cls if atom <= len(self.mol.GetAtoms())]
            ismiles = 0
            for x in fragall.split('.'):
                xmol = Chem.MolFromSmiles(x)
                atom_map = [atom.GetAtomMapNum() for atom in xmol.GetAtoms() if atom.GetSymbol() != '*']  
                if len(clss) != len(atom_map): continue
                tpot = 0          
                for j in clss :
                    if j not in atom_map :
                        tpot = 1
                        break
                if tpot == 0 :
                    ismiles = x
                    break
            if ismiles == 0 : print('can not get ismiles!',self.smiles)       

            svocablist = []
            for mk in self.matches :
                if mk[0] in cls:
                    dismiles = getsmileswithdummy(ismiles,mk[0])
                    dcoresmiles = get_context_env(Chem.MolFromSmiles(dismiles),self.radius)
                    dsmiles = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(dismiles)))
                    svocablist.append((dcoresmiles,dsmiles,dismiles,self.atomlabel[mk[0]],self.atomlabel[mk[1]]))
                if mk[1] in cls:
                    dismiles = getsmileswithdummy(ismiles,mk[1])
                    dcoresmiles = get_context_env(Chem.MolFromSmiles(dismiles),self.radius)
                    dsmiles = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(dismiles)))
                    svocablist.append((dcoresmiles, dsmiles,dismiles,self.atomlabel[mk[1]],self.atomlabel[mk[0]]))  

            interlabel = []
            inter_atom = []
            bondlist=[]
            iternn = 0
            coresmiles = 0
            bond_cls = -1
            if pa[i] >= 0 :
                for bondi, mk in enumerate(self.matches) :
                    if (mk[0] in cls) and (mk[1] in self.clusters[pa[i]]) :
                        inter_atom.append(mk[0])
                        ismiles = getsmileswithdummy(ismiles,mk[0])
                        coresmiles1 = get_context_env(Chem.MolFromSmiles(ismiles),self.radius)
                        coresmiles2 = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(ismiles)))
                        interlabel.append((self.atomlabel[mk[1]]+'-'+self.atomlabel[mk[0]],ismiles))
                        iternn = self.atomlabel[mk[1]]+'-'+self.atomlabel[mk[0]]
                        bondlist.append((self.atomlabel[mk[1]],self.atomlabel[mk[0]]))                       
                        break
                    elif (mk[1] in cls) and (mk[0] in self.clusters[pa[i]]) :
                        inter_atom.append(mk[1])
                        ismiles = getsmileswithdummy(ismiles,mk[1])
                        coresmiles1 = get_context_env(Chem.MolFromSmiles(ismiles),self.radius)
                        coresmiles2 = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(ismiles)))
                        interlabel.append((self.atomlabel[mk[0]]+'-'+self.atomlabel[mk[1]],ismiles))
                        iternn = self.atomlabel[mk[0]]+'-'+self.atomlabel[mk[1]]
                        bond_cls = bondi
                        bondlist.append((self.atomlabel[mk[0]],self.atomlabel[mk[1]]))   
                        break
            else :
                for bondi,mk in enumerate(self.matches) :
                    if mk[0] in cls:
                        inter_atom.append(mk[0])
                        ismiles = getsmileswithdummy(ismiles,mk[0])
                        coresmiles1 = get_context_env(Chem.MolFromSmiles(ismiles),self.radius)
                        coresmiles2 = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(ismiles)))
                        interlabel.append((self.atomlabel[mk[1]]+'-'+self.atomlabel[mk[0]],ismiles))
                        iternn = self.atomlabel[mk[1]]+'-'+self.atomlabel[mk[0]]
                        bondlist.append((self.atomlabel[mk[1]],self.atomlabel[mk[0]]))   
                        break
                    if mk[1] in cls:
                        inter_atom.append(mk[1])
                        ismiles = getsmileswithdummy(ismiles,mk[1])
                        coresmiles1 = get_context_env(Chem.MolFromSmiles(ismiles),self.radius)
                        coresmiles2 = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(ismiles)))
                        interlabel.append((self.atomlabel[mk[0]]+'-'+self.atomlabel[mk[1]],ismiles))
                        iternn = self.atomlabel[mk[0]]+'-'+self.atomlabel[mk[1]]
                        bond_cls = bondi
                        bondlist.append((self.atomlabel[mk[0]],self.atomlabel[mk[1]]))   
                        break  
                
            if bondi == -1 : print('bond is error')
            tree.nodes[i]['smiles'] = coresmiles
            tree.nodes[i]['inter_label'] = interlabel
            linkbond = self.atomlabel[self.matches[bond_cls][0]]+'-'+self.atomlabel[self.matches[bond_cls][1]]
            tree.nodes[i]['label'] = (coresmiles1,coresmiles2,ismiles,linkbond)
            clssb = [x-1 for x in clss]
            tree.nodes[i]['cluster'] = clssb 
            tree.nodes[i]['assm_cands'] = {}
            tree.nodes[i]['inter_bond'] = bondlist
            tree.nodes[i]['vocab'] = svocablist

            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
                hist = [a for c in prev_sib[i] for a in self.clusters[c] if a <= len(self.mol.GetAtoms())] 
                pa_cls = [a for a in self.clusters[ pa[i] ] if a <= len(self.mol.GetAtoms())]
                ass_node = []
                for h in self.matches :
                    for j in pa_cls :
                        if (j+1 == h[0]) and (h[0] not in ass_node):
                            tree.nodes[i]['assm_cands'][h[0]-1] = self.atomlabel[h[0]]+'-'+self.atomlabel[h[1]]
                        if (j+1 == h[1]) and (h[1] not in ass_node):
                            tree.nodes[i]['assm_cands'][h[1]-1] = self.atomlabel[h[1]]+'-'+self.atomlabel[h[0]]

                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atom:
                    for ch_atom in self.mol_graph[fa_atom-1]:
                        if ch_atom+1 in diff:
                            label = self.mol_graph[ch_atom][fa_atom-1]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.mol_graph[ch_atom][fa_atom-1]['label'] = (label, child_order)
        return order
       
    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())
           
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            a1id = a1.GetIdx()
            a2id = a2.GetIdx()            
            btype = HierGraph.BOND_LIST.index( bond.GetBondType() )
            graph[a1id][a2id]['label'] = btype
            graph[a2id][a1id]['label'] = btype

        return graph
    
    @staticmethod
    def tensorize(mol_batch, vocab, avocab, radius):
        mol_batch = [HierGraph(x,radius) for x in mol_batch]
        tree_tensors, tree_batchG = HierGraph.tensorize_fragment([x.mol_tree for x in mol_batch], vocab)
        graph_tensors, graph_batchG = HierGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = max( [len(c) for x in mol_batch for c in x.clusters] )
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            assup={}
            for x in attr['assm_cands'] :
                assup[add(x, offset)] = tree_batchG.nodes[v]['assm_cands'][x]
            tree_batchG.nodes[v]['assm_cands'] = assup
            cgraph[v, :len(cls)] = torch.IntTensor(cls)
        
        all_orders = []
        for i,hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)

        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders

    def tensorize_graph(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )
            
            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fnode[v] = vocab[attr]
                agraph.append([])       

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else :
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

    def tensorize_fragment(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        bfnode,bfmess = [None],[(0,0,0,0)] 
        bagraph,bbgraph = [[]], [[]] 
        bscope = []
        bedge_dict = {}
        ball_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )
            
            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fsmart = Chem.MolToSmarts(Chem.MolFromSmiles(attr[1]))
                fnode[v] = (vocab[(attr[0],fsmart,attr[2])][0],vocab[(attr[0],fsmart,attr[2])][1],vocab.get_bondid(attr[3]))
                agraph.append([])       

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else:
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)


class HierGraphwithFragment(object):

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 20

    def __init__(self, smiles, radius=2, flag=1):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.flag = flag
        self.radius = radius

        self.mol_graph = self.build_mol_graph()
        self.clusters, self.atom_cls, self.fragmol = self.find_clusters()
        self.mol_tree = self.tree_decomp()        
        self.order = self.label_tree()

    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        return new_atom

    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: 
            return [(0,)], [[0]]

        clusters = []

        matches, cls, clusters, atomlabel  = [],[],[], []

        for atom in mol.GetAtoms():
            cls.append(atom.GetIdx()+1)
        clusters.append((cls))
            
        Chem.Kekulize(mol)
        new_mol = Chem.RWMol(mol)
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx()+1)
            
        fragmol = new_mol
   
        if 1 not in clusters[0]: 
            for i,cls in enumerate(clusters):
                if 1 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    break

        atom_cls = [[] for i in range(len(self.mol.GetAtoms())+1)]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom_cls[atom].append(i)

        return clusters, atom_cls, fragmol

    def tree_decomp(self):
        clusters = self.clusters
        graph = nx.empty_graph( len(clusters) )
        for atom, nei_cls in enumerate(self.atom_cls):
            if len(nei_cls) <= 1: continue
            for i,c1 in enumerate(nei_cls):
                for c2 in nei_cls[i + 1:]:
                    inter = set(clusters[c1]) & set(clusters[c2])
                    graph.add_edge(c1, c2, weight = len(inter))

        n, m = len(graph.nodes), len(graph.edges)
        assert n - m <= 1 
        return graph if n - m == 1 else nx.maximum_spanning_tree(graph)

    def label_tree(self):
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa 
            sorted_child = sorted([ y for y in self.mol_tree[x] if y != fa ]) 
            for idx,y in enumerate(sorted_child):
                self.mol_tree[x][y]['label'] = 0 
                self.mol_tree[y][x]['label'] = idx + 1 
                prev_sib[y] = sorted_child[:idx] 
                prev_sib[y] += [x, fa] if fa >= 0 else [x]
                order.append( (x,y,1) )
                dfs(order, pa, prev_sib, y, x)
                order.append( (y,x,0) )

        order, pa = [], {}
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))]
        dfs(order, pa, prev_sib, 0, -1)

        order.append( (0, None, 0) ) 
                                        
        fragall=Chem.MolToSmiles(self.fragmol,True)
        fraglist = [Chem.MolFromSmiles(x) for x in fragall.split('.')]

        tree = self.mol_tree
        for i,cls in enumerate(self.clusters):
            clss = [atom for atom in cls]
            ismiles = self.smiles
            coresmiles1 = get_context_env(Chem.MolFromSmiles(ismiles),self.radius)
            coresmiles2 = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(ismiles)))
            tree.nodes[i]['label'] = (coresmiles1,coresmiles2,ismiles)
            clssb = [x-1 for x in clss]
            tree.nodes[i]['cluster'] = clssb 

            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: 
                hist = [a for c in prev_sib[i] for a in self.clusters[c] if a <= len(self.mol.GetAtoms())] 
                pa_cls = [a for a in self.clusters[ pa[i] ] if a <= len(self.mol.GetAtoms())]

                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atom:
                    for ch_atom in self.mol_graph[fa_atom-1]:
                        if ch_atom+1 in diff:
                            label = self.mol_graph[ch_atom][fa_atom-1]['label']
                            if type(label) is int: 
                                self.mol_graph[ch_atom][fa_atom-1]['label'] = (label, child_order)

        return order
       
    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())
           
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            a1id = a1.GetIdx()
            a2id = a2.GetIdx()            
            btype = HierGraphwithFragment.BOND_LIST.index( bond.GetBondType() )
            graph[a1id][a2id]['label'] = btype
            graph[a2id][a1id]['label'] = btype

        return graph
    
    @staticmethod
    def tensorize(mol_batch, vocab, avocab, radius):
        mol_batch = [HierGraphwithFragment(x,radius) for x in mol_batch]
        tree_tensors, tree_batchG = HierGraphwithFragment.tensorize_fragment([x.mol_tree for x in mol_batch], vocab)
        graph_tensors, graph_batchG = HierGraphwithFragment.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = max( [len(c) for x in mol_batch for c in x.clusters] )
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)
        
        all_orders = []
        for i,hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)

        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders

    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )
            
            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fnode[v] = vocab[attr]
                agraph.append([])       

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else :
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

    def tensorize_fragment(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        bfnode,bfmess = [None],[(0,0,0,0)] 
        bagraph,bbgraph = [[]], [[]] 
        bscope = []
        bedge_dict = {}
        ball_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )
            
            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fsmart = Chem.MolToSmarts(Chem.MolFromSmiles(attr[2]))
                fnode[v] = (vocab[(attr[0],fsmart,attr[2])][0],vocab[(attr[0],fsmart,attr[2])][1],0)
                agraph.append([])       

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else:
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)
