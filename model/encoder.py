import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from model.tools import *
from model.hier_graph import HierGraph
from model.rnn import LSTM

class CoreEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(CoreEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.W_o = nn.Sequential( 
                nn.Linear(node_fdim + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        if rnn_type == 'LSTM':
            self.rnn = LSTM(input_size, hidden_size, depth) 
        else:
            raise ValueError('unsupported rnn cell type ' + rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph):
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
        mask[0, 0] = 0 #first node is padding
        return node_hiddens * mask, h #return only the hidden state (different from IncMPNEncoder in LSTM case)

class HierEncoder(nn.Module):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(HierEncoder, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.atom_size = atom_size = avocab.size()
        self.bond_size = bond_size = len(HierGraph.BOND_LIST) + HierGraph.MAX_POS

        self.E_c = nn.Sequential(
                nn.Embedding(vocab.size()[0], embed_size),
                nn.Dropout(dropout)
        )
        self.E_i = nn.Sequential(
                nn.Embedding(vocab.size()[1], embed_size),
                nn.Dropout(dropout)
        )
        self.E_l = nn.Sequential(
                nn.Embedding(vocab.bondsize(), embed_size),
                nn.Dropout(dropout)
        )
        self.W_c = nn.Sequential( 
                nn.Linear(embed_size + hidden_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )
        self.W_i = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, embed_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        self.E_a = torch.eye(atom_size).cuda()
        self.E_b = torch.eye( len(HierGraph.BOND_LIST) ).cuda()
        self.E_apos = torch.eye( HierGraph.MAX_POS ).cuda()
        self.E_pos = torch.eye( HierGraph.MAX_POS ).cuda()

        self.W_root = nn.Sequential( 
                nn.Linear(hidden_size * 2, hidden_size), 
                nn.Tanh() #root activation is tanh
        )
        self.interchangeable_encoder = CoreEncoder(rnn_type, hidden_size + HierGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.frag_encoder = CoreEncoder(rnn_type, hidden_size + HierGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.bond_encoder = CoreEncoder(rnn_type, hidden_size + HierGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = CoreEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depthG, dropout)

    def tie_embedding(self, other):
        self.E_c, self.E_i = other.E_c, other.E_i
        self.E_a, self.E_b = other.E_a, other.E_b
    
    def embed_frag(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput1 = self.E_i(fnode[:, 1])

        hnode = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        hnode = self.W_i( torch.cat([finput1, hnode], dim=-1) )        

        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 

        return hnode, hmess, agraph, bgraph

    def embed_interchangeable(self, tree_tensors, hinter, hbond):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_c(fnode[:, 0])
        hnode = self.W_c( torch.cat([finput, hinter, hbond], dim=-1) )

        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        
        return hnode, hmess, agraph, bgraph

    def embed_bond(self, bond_tensors):
        fnode, fmess, agraph, bgraph, _ , _ = bond_tensors
        hnode = self.E_l(fnode[:, 2])
        
        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 

        return hnode, hmess, agraph, bgraph
    
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        hnode = self.E_a.index_select(index=fnode, dim=0)
        fmess1 = hnode.index_select(index=fmess[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0)
        fpos = self.E_apos.index_select(index=fmess[:, 3], dim=0)
        hmess = torch.cat([fmess1, fmess2, fpos], dim=-1)
        return hnode, hmess, agraph, bgraph

    def embed_root(self, hmess, tree_tensors, roots):
        roots = tree_tensors[2].new_tensor(roots) 
        fnode = tree_tensors[0].index_select(0, roots)
        agraph = tree_tensors[2].index_select(0, roots)

        nei_message = index_select_ND(hmess, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        return self.W_root(node_hiddens)

    def forward(self, tree_tensors, graph_tensors):
        tensors = self.embed_graph(graph_tensors)
        hatom,_ = self.graph_encoder(*tensors)

        tensors = self.embed_bond(tree_tensors)
        hbond,hmess = self.bond_encoder(*tensors)

        tensors = self.embed_frag(tree_tensors, hatom)
        hinter,hmess = self.frag_encoder(*tensors)

        tensors = self.embed_interchangeable(tree_tensors, hinter, hbond)
        hnode,hmess = self.interchangeable_encoder(*tensors)
        hroot = self.embed_root(hmess, tensors, [st for st,le in tree_tensors[-1]])

        return hroot, hnode, hinter, hbond, hatom

class FragEncoder(nn.Module):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(FragEncoder, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.atom_size = atom_size = avocab.size()
        self.bond_size = bond_size = len(HierGraph.BOND_LIST) + HierGraph.MAX_POS

        self.E_f = nn.Sequential(
                nn.Embedding(vocab.size()[0], embed_size),
                nn.Dropout(dropout)
        )

        self.W_f = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        self.E_a = torch.eye(atom_size).cuda()
        self.E_b = torch.eye( len(HierGraph.BOND_LIST) ).cuda()
        self.E_apos = torch.eye( HierGraph.MAX_POS ).cuda()
        self.E_pos = torch.eye( HierGraph.MAX_POS ).cuda()

        self.W_root = nn.Sequential( 
                nn.Linear(hidden_size * 2, hidden_size), 
                nn.Tanh() #root activation is tanh
        )
        self.interchangeable_encoder = CoreEncoder(rnn_type, hidden_size + HierGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = CoreEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depthG, dropout)

    def tie_embedding(self, other):
        self.E_c, self.E_i = other.E_c, other.E_i
        self.E_a, self.E_b = other.E_a, other.E_b
    

    def embed_interchangeable(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_f(fnode[:, 0])

        hnode = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        hnode = self.W_f( torch.cat([finput, hnode], dim=-1) )     

        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        
        return hnode, hmess, agraph, bgraph

    
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        hnode = self.E_a.index_select(index=fnode, dim=0)
        fmess1 = hnode.index_select(index=fmess[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0)
        fpos = self.E_apos.index_select(index=fmess[:, 3], dim=0)
        hmess = torch.cat([fmess1, fmess2, fpos], dim=-1)
        return hnode, hmess, agraph, bgraph

    def embed_root(self, hmess, tree_tensors, roots):
        roots = tree_tensors[2].new_tensor(roots) 
        fnode = tree_tensors[0].index_select(0, roots)
        agraph = tree_tensors[2].index_select(0, roots)

        nei_message = index_select_ND(hmess, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        return self.W_root(node_hiddens)

    def forward(self, tree_tensors, graph_tensors):
        tensors = self.embed_graph(graph_tensors)
        hatom,_ = self.graph_encoder(*tensors)

        tensors = self.embed_interchangeable(tree_tensors, hatom)
        hnode,hmess = self.interchangeable_encoder(*tensors)
        hroot = self.embed_root(hmess, tensors, [st for st,le in tree_tensors[-1]])

        return hroot, hnode, hatom

class MessEncoder(CoreEncoder):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(MessEncoder, self).__init__(rnn_type, input_size, node_fdim, hidden_size, depth, dropout)

    def forward(self, tensors, h, num_nodes, subset):
        fnode, fmess, agraph, bgraph = tensors
        subnode, submess = subset

        if len(submess) > 0: 
            h = self.rnn.sparse_forward(h, fmess, submess, bgraph)

        nei_message = index_select_ND(self.rnn.get_hidden_state(h), 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
        node_hiddens = index_scatter(node_hiddens, node_buf, subnode)
        return node_hiddens, h

class MessHierEncoder(HierEncoder):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(MessHierEncoder, self).__init__(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.interchangeable_encoder = MessEncoder(rnn_type, hidden_size + HierGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.frag_encoder = MessEncoder(rnn_type, hidden_size + HierGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.bond_encoder = MessEncoder(rnn_type, hidden_size + HierGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = MessEncoder(rnn_type, self.atom_size + self.bond_size, self.atom_size, hidden_size, depthG, dropout)
        del self.W_root

    def get_sub_tensor(self, tensors, subset):
        subnode, submess = subset
        fnode, fmess, agraph, bgraph = tensors[:4]
        subnode = subnode.cuda()
        fnode, fmess = fnode.index_select(0, subnode), fmess.index_select(0, submess)
        agraph, bgraph = agraph.index_select(0, subnode), bgraph.index_select(0, submess)

        if len(tensors) == 6:
            cgraph = tensors[4].index_select(0, subnode)
            return fnode, fmess, agraph, bgraph, cgraph, tensors[-1]
        else:
            return fnode, fmess, agraph, bgraph, tensors[-1]

    def embed_sub_bond(self, bond_tensors, subtree):
        subnode, submess = subtree
        num_nodes = bond_tensors[0].size(0)
        fnode, fmess, agraph, bgraph, cgraph, _ = self.get_sub_tensor(bond_tensors, subtree)

        hnode = self.E_l(fnode[:, 2])
            
        if len(submess) == 0:
            hmess = fmess
        else:
            node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
            node_buf = index_scatter(hnode, node_buf, subnode)
            hmess = node_buf.index_select(index=fmess[:, 0], dim=0)
            pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
            hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 

        return hnode, hmess, agraph, bgraph 

    def embed_sub_frag(self, frag_tensors, hgraph, subtree):
        subnode, submess = subtree
        num_nodes = frag_tensors[0].size(0)
        fnode, fmess, agraph, bgraph, cgraph, _ = self.get_sub_tensor(frag_tensors, subtree)

        finput = self.E_i(fnode[:, 1])
        hgraph = index_select_ND(hgraph, 0, cgraph).sum(dim=1)
        hnode = self.W_i( torch.cat([finput, hgraph], dim=-1) )

        if len(submess) == 0:
            hmess = fmess
        else:
            node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
            node_buf = index_scatter(hnode, node_buf, subnode)
            hmess = node_buf.index_select(index=fmess[:, 0], dim=0)
            pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
            hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 

        return hnode, hmess, agraph, bgraph 

    def embed_sub_interchangeable(self, interchangeable_tensors, hfrag, hbond, subtree):
        subnode, submess = subtree
        num_nodes = interchangeable_tensors[0].size(0)
        fnode, fmess, agraph, bgraph, cgraph, _ = self.get_sub_tensor(interchangeable_tensors, subtree)

        finput = self.E_c(fnode[:, 0])
        hinter = hfrag.index_select(0, subnode)
        hbond = hbond.index_select(0, subnode)
        hnode = self.W_c( torch.cat([finput, hinter, hbond], dim=-1) )
            
        if len(submess) == 0:
            hmess = fmess
        else:
            node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
            node_buf = index_scatter(hnode, node_buf, subnode)
            hmess = node_buf.index_select(index=fmess[:, 0], dim=0)
            pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
            hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 

        return hnode, hmess, agraph, bgraph 

    def forward(self, interchangeable_tensors, frag_tensors, bond_tensors, graph_tensors, hinterchangeable, hfrag, hbond, hgraph, subtree, subgraph):
        num_tree_nodes = interchangeable_tensors[0].size(0)
        num_graph_nodes = graph_tensors[0].size(0)

        if len(subgraph[0]) + len(subgraph[1]) > 0:
            sub_graph_tensors = self.get_sub_tensor(graph_tensors, subgraph)[:-1] #graph tensor is already embedded
            hgraph.node, hgraph.mess = self.graph_encoder(sub_graph_tensors, hgraph.mess, num_graph_nodes, subgraph)

        if len(subtree[0]) + len(subtree[1]) > 0:
            sub_frag_tensors = self.embed_sub_frag(frag_tensors, hgraph.node, subtree)
            hfrag.node, hfrag.mess = self.frag_encoder(sub_frag_tensors, hfrag.mess, num_tree_nodes, subtree)

            sub_bond_tensors = self.embed_sub_bond(bond_tensors, subtree)
            hbond.node, hbond.mess = self.bond_encoder(sub_bond_tensors, hbond.mess, num_tree_nodes, subtree)

            sub_interchangeable_tensors = self.embed_sub_interchangeable(interchangeable_tensors, hfrag.node, hbond.node, subtree)
            hinterchangeable.node, hinterchangeable.mess = self.interchangeable_encoder(sub_interchangeable_tensors, hinterchangeable.mess, num_tree_nodes, subtree)

        return hinterchangeable, hfrag, hbond, hgraph

