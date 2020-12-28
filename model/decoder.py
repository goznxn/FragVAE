import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
import numpy as np
import copy
from model.tools import *
from model.encoder import MessHierEncoder
from model.encoder import HierEncoder, FragEncoder
from model.hier_graph import HierGraph, HierGraphwithFragment
from model.mess_graph import IncTree, MessGraph
from model.dataset import *
from model.chemtools import *
from model.vocab import common_atom_vocab, DVocab

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c
    
class HTuple():
    def __init__(self, node=None, mess=None, vmask=None, emask=None):
        self.node, self.mess = node, mess
        self.vmask, self.emask = vmask, emask

class MultiVAE(nn.Module):
    def __init__(self, args):
        super(MultiVAE, self).__init__()
        self.encoder = HierEncoder(args.frag, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder = HierDecoder(args.frag, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.latent_size, args.diterT, args.diterG, args.dropout)

    def generator(self, fragsmiles, radius, batch_num, beam_size,topp, num_decode, enum_root, fragment_num, greedy=True):
        return self.decoder.beamgenerator(fragsmiles, radius, batch_num, beam_size, topp, num_decode, enum_root, fragment_num, greedy=greedy)
       
    def forward(self, frag_graphs, frag_tensors, mol_graphs, mol_tensors, orders, beta, perturb_z=True):
        loss, kl_div, topacc, clacc, iclacc, atacc = self.decoder(frag_graphs, frag_tensors, mol_graphs, mol_tensors, orders)
        return loss + beta * kl_div, kl_div.item(), topacc, clacc, iclacc, atacc

class HierDecoder(nn.Module):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, latent_size, depthT, depthG, dropout):
        super(HierDecoder, self).__init__()
        self.vocab = vocab
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.itensor = torch.LongTensor([]).cuda()

        self.encoder = HierEncoder(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.fragencoder = FragEncoder(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        
        self.hmpn = MessHierEncoder(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.rnn_cell = self.hmpn.interchangeable_encoder.rnn
        self.E_assm = self.hmpn.E_i 
        self.E_order = torch.eye(HierGraph.MAX_POS).cuda()

        self.R_mean = nn.Linear(hidden_size, latent_size)
        self.R_var = nn.Linear(hidden_size, latent_size)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)

        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

        #self.W_tree = nn.Sequential( nn.Linear(self.hidden_size + self.latent_size, self.hidden_size), nn.ReLU() )
        #self.W_graph = nn.Sequential( nn.Linear(self.hidden_size + self.latent_size, self.hidden_size), nn.ReLU() )
        self.W_root = nn.Sequential( nn.Linear(self.latent_size + self.latent_size, self.latent_size), nn.ReLU() )

        self.assmbondNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab.bondsize())
        )
        self.clsNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab.size()[0])
        )
        self.iclsNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab.size()[1])
        )
        self.matchNN = nn.Sequential(
                nn.Linear(hidden_size + embed_size + HierGraph.MAX_POS, hidden_size),
                nn.ReLU(),
        )
        self.W_assm = nn.Linear(hidden_size, latent_size)

        if latent_size != hidden_size:
            self.W_root = nn.Linear(latent_size, hidden_size)

        self.topo_loss = nn.BCEWithLogitsLoss(size_average=False)
        self.bond_loss = nn.CrossEntropyLoss(size_average=False)
        self.cls_loss = nn.CrossEntropyLoss(size_average=False)
        self.icls_loss = nn.CrossEntropyLoss(size_average=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        #z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def encode(self, tensors):
        tree_tensors, graph_tensors = tensors
        root_vecs, tree_vecs, _, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        #root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, True)

        tree_vecs = stack_pad_tensor( [tree_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        size = tree_vecs.new_tensor([le for _,le in tree_tensors[-1]])
        tree_vecs = tree_vecs.sum(dim=1) / size.unsqueeze(-1)
      
        graph_vecs = stack_pad_tensor( [graph_vecs[st : st + le] for st,le in graph_tensors[-1]] )
        size = graph_vecs.new_tensor([le for _,le in graph_tensors[-1]])
        graph_vecs = graph_vecs.sum(dim=1) / size.unsqueeze(-1)

        #tree_vecs, tree_kl = self.rsample(tree_vecs, self.R_mean, self.R_var, True)
        #graph_vecs, graph_kl = self.rsample(graph_vecs, self.R_mean, self.R_var, True)

        return root_vecs, tree_vecs, graph_vecs

    def fragencode(self, tensors):
        tree_tensors, graph_tensors = tensors
        root_vecs, tree_vecs, graph_vecs = self.fragencoder(tree_tensors, graph_tensors)
        #root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, True)

        return root_vecs, tree_vecs, graph_vecs
        
    def apply_tree_mask(self, tensors, cur, prev):
        fnode, fmess, agraph, bgraph, cgraph, scope = tensors
        agraph = agraph * index_select_ND(cur.emask, 0, agraph)
        bgraph = bgraph * index_select_ND(cur.emask, 0, bgraph)
        cgraph = cgraph * index_select_ND(prev.vmask, 0, cgraph)
        return fnode, fmess, agraph, bgraph, cgraph, scope

    def apply_graph_mask(self, tensors, hgraph):
        fnode, fmess, agraph, bgraph, scope = tensors
        agraph = agraph * index_select_ND(hgraph.emask, 0, agraph)
        bgraph = bgraph * index_select_ND(hgraph.emask, 0, bgraph)
        return fnode, fmess, agraph, bgraph, scope

    def update_graph_mask(self, graph_batch, new_atoms, hgraph):
        new_atom_index = hgraph.vmask.new_tensor(new_atoms)
        hgraph.vmask.scatter_(0, new_atom_index, 1)

        new_atom_set = set(new_atoms)
        new_bonds = [] #new bonds are the subgraph induced by new_atoms
        for zid in new_atoms:
            for nid in graph_batch[zid]:
                if nid not in new_atom_set: continue
                new_bonds.append( graph_batch[zid][nid]['mess_idx'] )

        new_bond_index = hgraph.emask.new_tensor(new_bonds)
        if len(new_bonds) > 0:
            hgraph.emask.scatter_(0, new_bond_index, 1)
        return new_atom_index, new_bond_index

    def init_decoder_state(self, tree_batch, tree_tensors, src_root_vecs):
        batch_size = len(src_root_vecs)
        num_mess = len(tree_tensors[1])
        agraph = tree_tensors[2].clone()
        bgraph = tree_tensors[3].clone()

        for i,tup in enumerate(tree_tensors[-1]):
            root = tup[0]
            assert agraph[root,-1].item() == 0
            agraph[root,-1] = num_mess + i
            for v in tree_batch.successors(root):
                mess_idx = tree_batch[root][v]['mess_idx'] 
                assert bgraph[mess_idx,-1].item() == 0
                bgraph[mess_idx,-1] = num_mess + i

        new_tree_tensors = tree_tensors[:2] + [agraph, bgraph] + tree_tensors[4:]
        htree = HTuple()
        htree.mess = self.rnn_cell.get_init_state(tree_tensors[1], src_root_vecs)
        htree.emask = torch.cat( [bgraph.new_zeros(num_mess), bgraph.new_ones(batch_size)], dim=0 )

        return htree, new_tree_tensors

    def get_bond_score(self, src_tree_vecs, batch_idx, cls_vecs, cls_labs):
        cls_cxt = src_tree_vecs.index_select(index=batch_idx, dim=0)
        cls_vecs = torch.cat([cls_vecs, cls_cxt], dim=-1)
        
        if cls_labs is None :
            bond_scores = self.assmbondNN(cls_vecs)
        else :
            bondmask = self.vocab.get_assmbond_mask(cls_labs)
            bond_scores = self.assmbondNN(cls_vecs) + bondmask

        return bond_scores

    def get_cls_score(self, src_tree_vecs, batch_idx, preassm, cls_vecs, cls_labs):
        cls_cxt = src_tree_vecs.index_select(index=batch_idx, dim=0)
        cls_vecs = torch.cat([cls_vecs, cls_cxt], dim=-1)
        
        if preassm is None :
            cls_scores = self.clsNN(cls_vecs)
        else :
            clsmask = self.vocab.get_cls_mask(preassm)
            cls_scores = self.clsNN(cls_vecs) + clsmask

        if cls_labs is None and preassm is None : #inference mode
            icls_scores = self.iclsNN(cls_vecs) #no masking
        if cls_labs is not None and preassm is None :
            vocab_masks = self.vocab.get_mask(cls_labs)
            icls_scores = self.iclsNN(cls_vecs) + vocab_masks 
        if cls_labs is not None and preassm is not None :
            vocab_masks = self.vocab.get_icls_mask(preassm,cls_labs)
            icls_scores = self.iclsNN(cls_vecs) + vocab_masks #apply mask by log(x + mask): mask=0 or -INF

        return cls_scores, icls_scores

    def get_assm_score(self, src_graph_vecs, batch_idx, assm_vecs):
        assm_cxt = index_select_ND(src_graph_vecs, 0, batch_idx)
        return (self.W_assm(assm_vecs) * assm_cxt).sum(dim=-1)

    def forward(self, frag_graphs, frag_tensors, mol_graphs, mol_tensors, orders):
        batch_size = len(orders)
        tree_batch, graph_batch = mol_graphs

        frag_tensors = make_cuda(frag_tensors)
        tree_tensors, graph_tensors = mol_tensors = make_cuda(mol_tensors)
        
        frag_root_vecs, frag_tree_vecs, frag_graph_vecs = self.fragencode(frag_tensors)
        mol_root_vecs, mol_tree_vecs, mol_graph_vecs = self.encode(mol_tensors)        
                                
        inter_tensors = tree_tensors
        bond_tensors = tree_tensors
        
        root_vecs, root_kl = self.rsample(mol_root_vecs, self.R_mean, self.R_var)
        kl_div = root_kl
        
        root_vecs = self.W_root( torch.cat([frag_root_vecs, root_vecs], dim=-1) )    
        src_tree_vecs = root_vecs 
        src_graph_vecs = root_vecs

        init_vecs = root_vecs if self.latent_size == self.hidden_size else self.W_root(root_vecs)

        htree, tree_tensors = self.init_decoder_state(tree_batch, tree_tensors, init_vecs)
        hinter = HTuple(
            mess = self.rnn_cell.get_init_state(inter_tensors[1]),
            emask = self.itensor.new_zeros(inter_tensors[1].size(0))
        )
        hbond = HTuple(
            mess = self.rnn_cell.get_init_state(bond_tensors[1]),
            emask = self.itensor.new_zeros(bond_tensors[1].size(0))
        )
        hgraph = HTuple(
            mess = self.rnn_cell.get_init_state(graph_tensors[1]),
            vmask = self.itensor.new_zeros(graph_tensors[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors[1].size(0))
        )

        all_topo_preds, all_cls_preds, all_assm_preds, all_pre_assm = [], [], [], []
        new_atoms = []
        tree_scope = tree_tensors[-1]
        for i in range(batch_size):
            root = tree_batch.nodes[ tree_scope[i][0] ]
            fsmart = Chem.MolToSmarts(Chem.MolFromSmiles(root['label'][1]))
            lab,ilab = self.vocab[ (root['label'][0] ,fsmart, root['label'][2]) ]
            linkbond = self.vocab.get_bondid(root['inter_label'][0][0])
            preassm = [ root['assm_cands'][i] for i in root['assm_cands']]
            all_cls_preds.append( (init_vecs[i], i, linkbond, lab, ilab) ) #cluster prediction
            all_pre_assm.append(preassm)
            new_atoms.extend(root['cluster'])

        subgraph = self.update_graph_mask(graph_batch, new_atoms, hgraph)
        graph_tensors = self.hmpn.embed_graph(graph_tensors) + (graph_tensors[-1],) #preprocess graph tensors

        maxt = max([len(x) for x in orders])
        max_cls_size = max( [len(attr) * 2 for node,attr in tree_batch.nodes(data='cluster')] )

        for t in range(maxt):
            batch_list = [i for i in range(batch_size) if t < len(orders[i])]
            assert htree.emask[0].item() == 0 and hgraph.vmask[0].item() == 0 and hgraph.emask[0].item() == 0

            subtree = [], []
            for i in batch_list:
                xid, yid, tlab = orders[i][t]
                subtree[0].append(xid)
                if yid is not None:
                    mess_idx = tree_batch[xid][yid]['mess_idx']
                    subtree[1].append(mess_idx)

            subtree = htree.emask.new_tensor(subtree[0]), htree.emask.new_tensor(subtree[1]) 
            htree.emask.scatter_(0, subtree[1], 1)
            hinter.emask.scatter_(0, subtree[1], 1)
            hbond.emask.scatter_(0, subtree[1], 1)

            cur_tree_tensors = self.apply_tree_mask(tree_tensors, htree, hgraph)
            cur_inter_tensors = self.apply_tree_mask(inter_tensors, hinter, hgraph)
            cur_bond_tensors = self.apply_tree_mask(bond_tensors, hbond, hgraph)
            cur_graph_tensors = self.apply_graph_mask(graph_tensors, hgraph)

            htree, hinter, hbond, hgraph = self.hmpn(cur_tree_tensors, cur_inter_tensors, cur_bond_tensors, cur_graph_tensors, htree, hinter, hbond, hgraph, subtree, subgraph)

            new_atoms = []
            for i in batch_list:
                xid, yid, tlab = orders[i][t]
                if yid is not None:
                    mess_idx = tree_batch[xid][yid]['mess_idx']
                    new_atoms.extend( tree_batch.nodes[yid]['cluster'] ) 

                if tlab == 0: continue

                cls = tree_batch.nodes[yid]['smiles']
                cls_label = tree_batch.nodes[yid]['label']
                cls_inter_label = tree_batch.nodes[yid]['inter_label']
                fnodesmart = Chem.MolToSmarts(Chem.MolFromSmiles(tree_batch.nodes[yid]['label'][1]))
                lab, ilab = self.vocab[ (tree_batch.nodes[yid]['label'][0], fnodesmart, tree_batch.nodes[yid]['label'][2]) ]
                treebond = self.vocab.get_bondid( tree_batch.nodes[yid]['inter_label'][0][0])
                preassm = [ tree_batch.nodes[yid]['assm_cands'][i] for i in tree_batch.nodes[yid]['assm_cands']]
                mess_idx = tree_batch[xid][yid]['mess_idx']
                hmess = self.rnn_cell.get_hidden_state(htree.mess)
                all_cls_preds.append( (hmess[mess_idx], i, treebond, lab, ilab) ) 
                all_pre_assm.append(preassm)

                cls_bonds = self.vocab.get_frag_bondid(cls_label)

                assm_pred = [] 
                for assm in tree_batch.nodes[yid]['assm_cands'] :
                    asbond = tree_batch.nodes[yid]['assm_cands'][assm]
                    asbond = asbond.split('-')
                    asbond = asbond[1]+'-'+asbond[0]
                    bondid = self.vocab.get_bondid(asbond)
                    if bondid in cls_bonds :
                        assm_pred.append((assm,bondid,ilab))

                if len(assm_pred) > 1 :
                    nth_child = tree_batch[yid][xid]['label'] 
                    cands = [i[0] for i in assm_pred]

                    cand_vecs = self.enum_attach(hgraph, cands, [ilab], nth_child)
                    if len(cand_vecs) < max_cls_size:
                        pad_len = max_cls_size - len(cand_vecs)
                        cand_vecs = F.pad(cand_vecs, (0,0,0,pad_len))

                    batch_idx = hgraph.emask.new_tensor( [i] * max_cls_size )
                    all_assm_preds.append( (cand_vecs, batch_idx, 0) )        

            subgraph = self.update_graph_mask(graph_batch, new_atoms, hgraph)

        cls_vecs, batch_idx, bondlist, cls_labs, icls_labs = zip_tensors(all_cls_preds)      
        bond_scores = self.get_bond_score(src_tree_vecs, batch_idx, cls_vecs, cls_labs)
        cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, batch_idx, None, cls_vecs, cls_labs)
        bond_loss = self.bond_loss(bond_scores,bondlist)
        cls_loss = self.cls_loss(cls_scores, cls_labs) 
        icls_loss = self.icls_loss(icls_scores, icls_labs)
        bond_acc = get_accuracy(bond_scores,bondlist)
        cls_acc = get_accuracy(cls_scores, cls_labs)
        icls_acc = get_accuracy(icls_scores, icls_labs)
        
        if len(all_assm_preds) > 0:
            assm_vecs, batch_idx, assm_labels = zip_tensors(all_assm_preds)
            assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, assm_vecs)
            assm_loss = self.assm_loss(assm_scores, assm_labels)
            assm_acc = get_accuracy_sym(assm_scores, assm_labels)
        else:
            assm_loss, assm_acc = 0, 1

        loss = (cls_loss +icls_loss  + assm_loss + bond_loss) / batch_size
        return loss, kl_div, bond_acc, cls_acc, icls_acc, assm_acc

    def enum_attach(self, hgraph, cands, icls, nth_child):
        cands = self.itensor.new_tensor(cands)
        icls_vecs = self.itensor.new_tensor(icls * len(cands) )
        icls_vecs = self.E_assm( icls_vecs )

        nth_child = self.itensor.new_tensor([nth_child] * len(cands.view(-1)))
        order_vecs = self.E_order.index_select(0, nth_child)
        cand_vecs = hgraph.node.index_select(0, cands.view(-1))
        cand_vecs = torch.cat( [cand_vecs, icls_vecs, order_vecs], dim=-1 )
        cand_vecs = self.matchNN(cand_vecs)

        return cand_vecs

    def singleBeamdecode(self, fragsmiles, batch_num, beam_size, topp, num_decode, enum_root, fragment_num, greedy=True, max_decode_step=10, beam=20):
        fragtensor = HierGraphwithFragment.tensorize([fragsmiles], self.vocab, self.avocab)
        fraggraph, fragtensor, order = to_numpy(fragtensor)
        fragtensor = make_cuda(fragtensor)

        frag_root_vecs, frag_tree_vecs, frag_graph_vecs = self.fragencode(fragtensor)
        frag_root_vecs = frag_root_vecs.expand(num_decode, self.latent_size)
        
        batch_size = batch_num
        cur_decode_num = num_decode - batch_size
        cur_stack_num = batch_size

        src_root_vecs = torch.randn(num_decode, self.latent_size).cuda()

        src_root_vecs = self.W_root( torch.cat([frag_root_vecs, src_root_vecs], dim=-1) )    
        #z_graph = torch.randn(batch_size, self.latent_size).cuda()
        src_tree_vecs = src_root_vecs 
        src_graph_vecs =src_root_vecs

        tree_batch = IncTree(int(num_decode), node_fdim=3, edge_fdim=4)
        graph_batch = IncGraph(self.avocab, num_decode, node_fdim=self.hmpn.atom_size, edge_fdim=self.hmpn.atom_size + self.hmpn.bond_size)

        stack = [[] for i in range(num_decode)]
        fragnum = [0 for i in range(num_decode)]

        #init_vecs = src_root_vecs if self.latent_size == self.hidden_size else self.W_root(src_root_vecs)
        batch_idx = self.itensor.new_tensor(range(num_decode))
        #cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, batch_idx, None, src_tree_vecs, None)
        #cls_scores, icls_scores = self.get_cls_score(tree_vecs, batch_idx, init_vecs, None)
        #root_cls = cls_scores.max(dim=-1)[1]
        #icls_scores = icls_scores + self.vocab.get_mask(root_cls)
        #root_cls, root_icls = root_cls.tolist(), icls_scores.max(dim=-1)[1].tolist()
        #root_cls, root_icls = root_cls_topP_radom(cls_scores, icls_scores,self.vocab, batch_size)
        #root_cls, root_icls = root_cls_topP_Probability(cls_scores, icls_scores,self.vocab, batch_size)

        super_root = tree_batch.add_node() 
        for bid in range(batch_size):
            coresmiles1 = get_context_env(Chem.MolFromSmiles(fragsmiles),2)
            coresmiles2 = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(fragsmiles)))
            clab, ilab = self.vocab[ (coresmiles1 ,coresmiles2, fragsmiles) ]
            
            root_idx = tree_batch.add_node( batch_idx.new_tensor([0, 0, 0]) )
            tree_batch.add_edge(super_root, root_idx,) 

            #cursmiles = self.vocab.get_ismiles(ilab)
            #curcore, cursmiles = self.vocab.get_smiles(clab), self.vocab.get_ismiles(ilab)
            #print('root smiles',cursmiles)
            curcore = self.vocab.get_smiles(clab)
            cursmiles = fragsmiles
            new_atoms, new_bonds, attached = graph_batch.add_mol(bid, cursmiles, [], 0, curcore, clab)
            tree_batch.register_cgraph(root_idx,cursmiles, new_atoms, new_bonds, attached)
            #print('root smiles',cursmiles)
            root_cluster, _, root_used = tree_batch.get_cluster(root_idx)
            cands = [ [x] for x in root_cluster if x[0] not in root_used ]        
            stack[bid].append([root_idx,len(cands),super_root,0])
            fragnum[bid] = 1

        #invariance: tree_tensors is equal to inter_tensors (but inter_tensor's init_vec is 0)
        tree_tensors = tree_batch.get_tensors()
        graph_tensors = graph_batch.get_tensors()

        htree = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
        hinter = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
        hbond = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
        hgraph = HTuple( mess = self.rnn_cell.get_init_state(graph_tensors[1]) )
        #h = self.rnn_cell.get_hidden_state(htree.mess)
        #h[1 : batch_size + 1] = src_root_vecs #wiring root (only for tree, not inter)
        
        for t in range(max_decode_step):
            new_mess = []       
            for bid in range(num_decode) :
                snum = len(stack[bid])
                for i in range(snum) :
                    if stack[bid][-1][1] <= 0 :
                        if stack[bid][-1][2] > super_root :
                            nth_child = tree_batch.graph.in_degree(stack[bid][-1][2]) #edge child -> father has not established
                            edge_feature = batch_idx.new_tensor( [stack[bid][-1][0], stack[bid][-1][2], nth_child, stack[bid][-1][3]] )
                            new_edge = tree_batch.add_edge(stack[bid][-1][0], stack[bid][-1][2], edge_feature)
                            #new_mess.append(new_edge)
                        if stack[bid][-1][1] == -1 :
                            for node in range(len(stack[bid])) :
                                if stack[bid][node][0] == stack[bid][-1][2] :
                                    stack[bid][node][1] = stack[bid][node][1] - 1
                                    break
                                                                
                        child = stack[bid].pop()
            batch_list = [ bid for bid in range(num_decode) if len(stack[bid]) > 0 ]
            fflist = [ bid for bid in batch_list if fragnum[bid]< fragment_num ]
            if len(batch_list) == 0 or len(fflist) == 0 : break

            #src_tree_vecs = self.W_tree( torch.cat([tree_vecs, z_tree], dim=-1) )
            #src_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph], dim=-1) )

            batch_idx = batch_idx.new_tensor(batch_list)
            cur_tree_nodes = [stack[bid][-1][0] for bid in batch_list]
            subtree = batch_idx.new_tensor(cur_tree_nodes), batch_idx.new_tensor([])
            subgraph = batch_idx.new_tensor( tree_batch.get_cluster_nodes(cur_tree_nodes) ), batch_idx.new_tensor( tree_batch.get_cluster_edges(cur_tree_nodes) )

            expand_list = []
            for i,bid in enumerate(batch_list):
                if len(stack[bid]) > 0 and fragnum[bid] < fragment_num :
                    expand_list.append( (len(new_mess), bid) )
                    new_node = tree_batch.add_node() #new node label is yet to be predicted
                    edge_feature = batch_idx.new_tensor( [stack[bid][-1][0], new_node, 0, 0] ) #parent to child is 0
                    new_edge = tree_batch.add_edge(stack[bid][-1][0], new_node, edge_feature) 
                    stack[bid].append([new_node,-1,stack[bid][-1][0], 0])
                    new_mess.append(new_edge)

            subtree = subtree[0], batch_idx.new_tensor(new_mess)
            htree, hinter, hbond, hgraph = self.hmpn(tree_tensors, tree_tensors, tree_tensors, graph_tensors, htree, hinter, hbond, hgraph, subtree, subgraph)
            cur_mess = self.rnn_cell.get_hidden_state(htree.mess).index_select(0, subtree[1])
                   
            if len(expand_list) > 0:
                idx_in_mess, expand_list = zip(*expand_list)
                #print('idx_in_mess',idx_in_mess)
                idx_in_mess = batch_idx.new_tensor( idx_in_mess )
                expand_idx = batch_idx.new_tensor( expand_list )
                forward_mess = cur_mess.index_select(0, idx_in_mess)
                bond_scores = self.get_bond_score(src_tree_vecs, expand_idx, forward_mess, None)
                cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, expand_idx, None, forward_mess, None)
                #print('src_tree_vecs, expand_idx  forward_mess bond_scores shape',src_tree_vecs.shape,expand_idx.shape,forward_mess.shape,bond_scores.shape)
                #scores, cls_topk, icls_topk = hier_topk(cls_scores,icls_scores, self.vocab, beam)
                #if not greedy:
                #    scores = torch.exp(scores) #score is output of log_softmax
                #    shuf_idx = torch.multinomial(scores, beam, replacement=True).tolist()      

            for i,bid in enumerate(expand_list):   
                #print(' expand_list',expand_list,i,bid)  
                fa_node = stack[bid][-2][0]
                addbond = []
                #nodesource = stack[bid][-1]
                #new_node = nodesource[0]
                #fa_node = nodesource[2]
                #print('stack[bid][-1]',stack[bid][-1])
                bidaddlist = []
                bidatommap = {}

                for kk in range(beam):       
                    new_node = stack[bid][-1][0]
                    fa_cluster, _, fa_used = tree_batch.get_cluster(fa_node)
                    fa_ismiles = tree_batch.get_smiles(fa_node)
                    
                    #cands = [x[0] for x in fa_cluster if x[0] not in fa_used ]
                    cands = [x[0] for x in fa_used if x[1] > 0 ]

                    firstsign = 0
                    if len(cands) == 0  :  break
                    #fa_assm_atom = [x for x in fa_cluster if x[0] in cands]   
                    #fa_assm_symbol = [x[1] for x in fa_cluster if x[0] in cands]
                    fa_assm_atom = [x for x in fa_cluster if x[0] == cands[0] ]  
                    fa_assm_symbol = [x[1] for x in fa_cluster if x[0] == cands[0]]
                    fa_smiles = Chem.MolToSmiles(set_atommap(Chem.MolFromSmiles(fa_ismiles)))
                    fa_bonds = []
                    fa_bonds = self.vocab.get_frag_bonds(fa_smiles) 
                    #if len(fa_bonds) == 0 : print('can not get bond!',fa_ismiles, fa_smiles)
                    
                    if len(fa_bonds) == 0 :
                        for x in fa_assm_symbol :
                            fa_bonds.extend(self.vocab.get_bondwithatom(x))
                    fa_bonds = list(set(fa_bonds))
                    fa_assm_bonds = [x for x in fa_bonds if x[0] in fa_assm_symbol]

                    bond_topk = hier_bond_topP(bond_scores[i], self.vocab, fa_assm_bonds, topp)  

                    #if len(bond_topk) < len(bidaddlist) + 1 : continue
                    #print(' bond_topk',bid,len(bond_topk),fa_assm_bonds)  
                    #bond_sorted_cands = sorted( list(zip(bond_topk[0], cur_bond_scores[0])), key = lambda x:x[1], reverse=True )
                    #for bond_label in bond_topk:
                    if len(cands) > 0 :
                        #if bond_label in addbond : continue
                        #addbond.append(bond_label) 
                        #cls_topk, icls_topk = hier_cls_topk(cls_scores[i],icls_scores[i], self.vocab, self.vocab.get_bond(bond_label),beam_size )
                        cls_topk, icls_topk = hier_cls_topP_Radom(cls_scores[i],icls_scores[i], self.vocab, bond_topk, beam_size, topp )
                        if len(cls_topk) < len(bidaddlist) + 1 : continue
 
                        for samtop in range(len(cls_topk)) :
                            clab, ilab = cls_topk[samtop], icls_topk[samtop]
                            #if ilab in addmol : continue
                            smiles, ismiles = self.vocab.get_smiles(clab), self.vocab.get_ismiles(ilab)
                            #print('clab,ilab',clab,ilab)
                            #print('clab,ilab smiles',self.vocab((smiles, ismiles)))
                            #curbond = self.vocab.get_bond(bond_label)
                            bond_label, curbond = self.vocab.get_bondwithdummy(fa_assm_symbol,ismiles)
                            node_feature = batch_idx.new_tensor( [clab,ilab,bond_label] )
                            tree_batch.set_node_feature(new_node, node_feature)
                            if curbond == 0 : print('bond error!')
                            #cur_assm = [x[0] for x in fa_assm_atom if x[1] == curbond[0]]  
                            cur_assm = [fa_assm_atom[0][0]]

                            if len(cur_assm) == 0:
                                continue
                            elif len(cur_assm) == 1:
                                #sorted_cands = [(cur_assm[0], curbond)]
                                inter_label = cur_assm[0]
                                nth_child = 0
                            else:
                                nth_child = tree_batch.graph.in_degree(fa_node)
                                cand_vecs = self.enum_attach(hgraph, cur_assm, [ilab], nth_child)
                                batch_idx = batch_idx.new_tensor( [bid] * len(cur_assm) )
                                assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, cand_vecs).tolist()
                                sorted_cands = sorted( list(zip(cur_assm, assm_scores)), key = lambda x:x[1], reverse=True )
                                inter_label = sorted_cands[0][0]
                                #for inter_label,_ in sorted_cands:
                            inter_label = list(zip([inter_label], [curbond]))
                                    #if graph_batch.try_add_mol(bid, ismiles, inter_label):
                            
                            if cur_decode_num > 0 and len(bidaddlist) == 0 and beam_size > 0 :  
                                if beam_size > len(cls_topk) -1 :
                                    cur_beam_add = len(cls_topk) -1
                                else :
                                    cur_beam_add = beam_size
                                if cur_decode_num > cur_beam_add :
                                    stacknewnum = cur_beam_add
                                    cur_decode_num = cur_decode_num - cur_beam_add
                                else :
                                    stacknewnum = cur_decode_num
                                    cur_decode_num = 0
                                
                                for sstnum in range(stacknewnum) :
                                    new_node_1 = tree_batch.add_node() #new node label is yet to be predicted
                                    edge_feature = batch_idx.new_tensor( [fa_node, new_node_1, 0, 0] ) #parent to child is 0
                                    new_edge = tree_batch.add_edge(fa_node, new_node_1, edge_feature) 
                                    stack[cur_stack_num] = copy.deepcopy(stack[bid])
                                    stack[cur_stack_num].pop()
                                    stack[cur_stack_num].append([new_node_1,-1,fa_node,0])
                                    bidaddlist.append(cur_stack_num)
                                    fragnum[cur_stack_num] = fragnum[bid]
                                    #new_mess.append(new_edge) 
                                    cur_stack_num = cur_stack_num + 1
                                    
                            if len(bidaddlist) > samtop:    
                                bidbeamid = bidaddlist[samtop]
                                new_node_beam = stack[bidbeamid][-1][0]
                                if len(bidatommap) > samtop :
                                    for x,y in inter_label :
                                        newattach = bidatommap[bidbeamid][x]
                                        bid_inter_label = list(zip([newattach], [curbond]))
                                else :
                                    bid_inter_label = inter_label
   
                                new_atoms, new_bonds, attached, atommap = graph_batch.add_beam_mol(bidbeamid, bid, ismiles, bid_inter_label, nth_child, smiles, clab)
                                if len(bidatommap) <= samtop : bidatommap[bidbeamid]=atommap
                                tree_batch.register_cgraph(new_node_beam, ismiles, new_atoms, new_bonds, attached)

                                fragnum[bidbeamid] = fragnum[bidbeamid]+1
                                fanodeid = 0
                                for node in range(len(stack[bidbeamid])) :
                                    if stack[bidbeamid][node][0] == fa_node :
                                        stack[bidbeamid][node][1] = stack[bidbeamid][node][1] - 1
                                        fanodeid = node
                                        break
     
                                cur_beam_cluster, _, cur_beam_used = tree_batch.get_cluster(new_node_beam)
                                cur_beam_cands = [ [x] for x in cur_beam_cluster if x[0] not in cur_beam_used ]
                                stack[bidbeamid][-1][1] = len(cur_beam_cands)
                                stack[bidbeamid][-1][3] = bond_label
                                edge_feature = batch_idx.new_tensor( [fa_node, new_node_beam, 0, bond_label] ) #parent to child is 0
                                tree_batch.update_edge_feature(fa_node, new_node_beam, edge_feature) 

                                #print('beam bid newbid inter_label ismiles fa_node curnode',bidbeamid, bid, inter_label, ismiles, stack[bidbeamid][fanodeid][1], len(cur_beam_cands))                                      
                                if stack[bidbeamid][fanodeid][1] > 0 :
                                    new_node_1 = tree_batch.add_node() #new node label is yet to be predicted
                                    edge_feature = batch_idx.new_tensor( [fa_node, new_node_beam, 0, 0] ) #parent to child is 0
                                    new_edge = tree_batch.add_edge(fa_node, new_node_beam, edge_feature) 
                                    stack[bidbeamid].append([new_node_beam,-1,fa_node,0])
                                    #new_mess.append(new_edge)                                               
                            else :                          
                                new_atoms, new_bonds, attached = graph_batch.add_mol(bid, ismiles, inter_label, nth_child, smiles, clab)
                                tree_batch.register_cgraph(new_node, ismiles, new_atoms, new_bonds, attached)
                                tree_batch.update_attached(fa_node, inter_label)

                                fragnum[bid] = fragnum[bid]+1
                                fanodeid = 0
                                for node in range(len(stack[bid])) :
                                    if stack[bid][node][0] == fa_node :
                                        stack[bid][node][1] = stack[bid][node][1] - 1
                                        fanodeid = node
                                        break

                                cur_cluster, _, cur_used = tree_batch.get_cluster(new_node)
                                cur_cands = [ [x] for x in cur_cluster if x[0] not in cur_used ]
                                stack[bid][-1][1] = len(cur_cands)
                                stack[bid][-1][3] = bond_label
                                edge_feature = batch_idx.new_tensor( [fa_node, new_node, 0, bond_label] ) #parent to child is 0
                                tree_batch.update_edge_feature(fa_node, new_node, edge_feature) 
                                #print('normal bid inter_label ismiles fa_node curnode', bid, inter_label, ismiles, stack[bid][fanodeid][1], len(cur_cands))                                                           
                                if stack[bid][fanodeid][1] > 0 :
                                    new_node_1 = tree_batch.add_node() #new node label is yet to be predicted
                                    edge_feature = batch_idx.new_tensor( [fa_node, new_node_1, 0, 0] ) #parent to child is 0
                                    new_edge = tree_batch.add_edge(fa_node, new_node_1, edge_feature) 
                                    stack[bid].append([new_node_1,-1,fa_node,0])
                                    #new_mess.append(new_edge)                                  
                                break 

                        break

        return graph_batch.get_mol()
