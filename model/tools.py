import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def avg_pool(all_vecs, scope, dim):
    size = create_var(torch.Tensor([le for _,le in scope]))
    return all_vecs.sum(dim=dim) / size.unsqueeze(-1)

def get_accuracy_bin(scores, labels):
    preds = torch.ge(scores, 0).long()
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()

def get_accuracy(scores, labels):
    _,preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()

def get_accuracy_sym(scores, labels):
    max_scores,max_idx = torch.max(scores, dim=-1)
    lab_scores = scores[torch.arange(len(scores)), labels]
    acc = torch.eq(lab_scores, max_scores).float()
    return torch.sum(acc) / labels.nelement()

def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i,tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad( tensor, (0,0,0,pad_len) )
    return torch.stack(tensor_list, dim=0)

def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)

def zip_tensors(tup_list):
    res = []
    tup_list = zip(*tup_list)
    for a in tup_list:
        if type(a[0]) is int: 
            res.append( torch.LongTensor(a).cuda() )
        else:
            res.append( torch.stack(a, dim=0) )
    return res

def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf

def hier_bond_topP(bond_scores, vocab, assmbond, topp):
    bond_mask = vocab.get_curbond_mask(assmbond)   
    #if len(assmbond[0]) == 1:
    #    topk = 1 
    #else :
    #    topk = len(assmbond[0])//2 + 1
    masked_bond_scores = F.log_softmax(bond_scores + bond_mask, dim=-1)
    temperature = 1
    top_p = topp
 
    masked_bond_scores = masked_bond_scores[-1, :] / temperature
    filtered_logits = top_p_filtering(masked_bond_scores, top_p=top_p)
    probabilities = F.softmax(filtered_logits, dim=-1)

    nup_mask = bond_mask + 10000
    maskvalid = len(torch.nonzero(nup_mask))
    
    numk = len(torch.nonzero(probabilities))

    if numk > maskvalid :  numk = maskvalid

    bond_scores_topk, bond_topk = probabilities.topk(numk, dim=-1)   
    bond_topk = np.array(bond_topk.cpu())
    
    #if len(bond_topk) > 1 :   random.shuffle(bond_topk)
    #bond_topk = torch.from_numpy(bond_topk).cuda()

    return bond_topk

def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):

    assert logits.dim() == 1  

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p    
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]  
    logits[indices_to_remove] = filter_value
    return logits


def hier_cls_topk(cls_scores, icls_scores, vocab, assmbond, beam_size):
    batch_size = len(cls_scores)
    cls_list, icls_list = [], []
 
    cls_mask = vocab.get_cls_mask(assmbond)
    masked_cls_scores = F.log_softmax(cls_scores + cls_mask, dim=-1)

    temperature = 1
    top_p = 0.95
 
    masked_cls_scores = masked_cls_scores[-1, :] / temperature
    filtered_logits = top_p_filtering(masked_cls_scores, top_p=top_p)
    probabilities = F.softmax(filtered_logits, dim=-1)
    #nup_mask = np.array(cls_mask.tolist())
    #maskvalid = int(sum(nup_mask[0]==0))
    nup_mask = cls_mask + 10000
    maskvalid = len(torch.nonzero(nup_mask))
    
    #if topk > beam_size+1 : topk = beam_size+1
    
    #pronum = np.array(probabilities.cpu())
    #numk = int(sum(pronum>0))
    numk = len(torch.nonzero(probabilities))
    
    topk = beam_size + 1
    if topk > maskvalid : topk = maskvalid
    if topk > numk : topk = numk
    #if numk > maskvalid : numk = maskvalid
    
    #topk = numk
    samnum = numk
    if samnum > maskvalid : samnum = maskvalid
    #cls_topk = torch.multinomial(probabilities, topk, replacement=False)
    #cls_scores_topk, cls_topk = probabilities.topk(samnum, dim=-1)  
    cls_scores_topk, cls_topk = probabilities.topk(topk, dim=-1)  
    #cls_scores_topk, cls_topk = masked_cls_scores.topk(topk, dim=-1)  
    fcls_topk = cls_topk
    #cls_topk = list(cls_topk.tolist())
    #fcls_topk = random.sample(cls_topk, topk)

    #fcls_topk = np.array(fcls_topk)
    #fcls_topk = torch.from_numpy(fcls_topk).cuda()


    final_topk = []
    for i in range(topk):
        clab = fcls_topk[i]
        icls_mask = vocab.get_icls_mask(assmbond,clab)
        #icls_mask = vocab.get_mask(clab)
        	
        masked_icls_scores = F.log_softmax(icls_scores + icls_mask, dim=-1)
        #icls_scores_topk, icls_topk = masked_icls_scores.topk(1, dim=-1)
        
        masked_icls_scores = masked_icls_scores[-1, :] / temperature
        icls_filtered_logits = top_p_filtering(masked_icls_scores, top_p=top_p)
        icls_probabilities = F.softmax(icls_filtered_logits, dim=-1)
            
        #icls_pronum = np.array(icls_probabilities.cpu())
        #icls_numk = int(sum(icls_probabilities>0))
        icls_numk = len(torch.nonzero(icls_probabilities))
        #icls_topk = torch.multinomial(icls_probabilities, 1, replacement=False)
        #icls_scores_topk, icls_topk = icls_probabilities.topk(icls_numk, dim=-1)
        icls_scores_topk, icls_topk = icls_probabilities.topk(1, dim=-1)
        icls_topk = list(icls_topk.tolist())
        #icls_topk = random.sample(icls_topk, 1)

        cls_list.append(clab.cpu())
        #icls_list.append(icls_topk[0].cpu())
        icls_list.append(icls_topk[0])
 
        #print('clab iclab',clab,icls_topk[0][0].cpu())
        #topk_scores = cls_scores_topk[0][i].unsqueeze(-1) + icls_scores_topk
        #final_topk.append( (topk_scores, clab.unsqueeze(-1).expand(-1, 1), icls_topk) )
        #final_topk.append( (topk_scores, clab.unsqueeze(-1).expand(-1, 1)) )

    #topk_scores, cls_topk, icls_topk = zip(*final_topk)
    #topk_scores = torch.cat(topk_scores, dim=-1)
    #cls_topk = torch.cat(cls_topk, dim=-1)
    #icls_topk = torch.cat(icls_topk, dim=-1)

    #topk_scores, topk_index = topk_scores.topk(topk, dim=-1)
    #batch_index = cls_topk.new_tensor([[i] * topk for i in range(batch_size)])
    #cls_topk = cls_topk[batch_index, topk_index]
    #icls_topk = icls_topk[batch_index, topk_index]

    return cls_list, icls_list

def hier_cls_topP_Radom(cls_scores, icls_scores, vocab, assmbond, beam_size, topp):
    batch_size = len(cls_scores)
    cls_list, icls_list = [], []
 
    cls_mask = vocab.get_cls_mask(assmbond)
    masked_cls_scores = F.log_softmax(cls_scores + cls_mask, dim=-1)

    temperature = 1
    top_p = topp
 
    masked_cls_scores = masked_cls_scores[-1, :] / temperature
    filtered_logits = top_p_filtering(masked_cls_scores, top_p=top_p)
    probabilities = F.softmax(filtered_logits, dim=-1)

    nup_mask = cls_mask + 10000
    maskvalid = len(torch.nonzero(nup_mask))
    
    numk = len(torch.nonzero(probabilities))
    
    topk = beam_size + 1
    if topk > maskvalid : topk = maskvalid
    if topk > numk : topk = numk
    
    samnum = numk
    if samnum > maskvalid : samnum = maskvalid

    cls_scores_topk, cls_topk = probabilities.topk(samnum, dim=-1)  

    cls_topk = list(cls_topk.tolist())
    fcls_topk = random.sample(cls_topk, topk)

    fcls_topk = np.array(fcls_topk)
    fcls_topk = torch.from_numpy(fcls_topk).cuda()

    final_topk = []
    for i in range(topk):
        clab = fcls_topk[i]
        icls_mask = vocab.get_icls_mask(assmbond,clab)
        	
        masked_icls_scores = F.log_softmax(icls_scores + icls_mask, dim=-1)
        
        masked_icls_scores = masked_icls_scores[-1, :] / temperature
        icls_filtered_logits = top_p_filtering(masked_icls_scores, top_p=top_p)
        icls_probabilities = F.softmax(icls_filtered_logits, dim=-1)
            
        icls_numk = len(torch.nonzero(icls_probabilities))
        icls_scores_topk, icls_topk = icls_probabilities.topk(icls_numk, dim=-1)

        icls_topk = list(icls_topk.tolist())
        icls_topk = random.sample(icls_topk, 1)

        cls_list.append(clab.cpu())
        icls_list.append(icls_topk[0])
 
    return cls_list, icls_list

def hier_cls_topP_probability(cls_scores, icls_scores, vocab, assmbond, beam_size, topp):
    batch_size = len(cls_scores)
    cls_list, icls_list = [], []
 
    cls_mask = vocab.get_cls_mask(assmbond)
    masked_cls_scores = F.log_softmax(cls_scores + cls_mask, dim=-1)

    temperature = 1
    top_p = topp
 
    masked_cls_scores = masked_cls_scores[-1, :] / temperature
    filtered_logits = top_p_filtering(masked_cls_scores, top_p=top_p)
    probabilities = F.softmax(filtered_logits, dim=-1)

    nup_mask = cls_mask + 10000
    maskvalid = len(torch.nonzero(nup_mask))
    
    numk = len(torch.nonzero(probabilities))
    
    topk = beam_size + 1
    if topk > maskvalid : topk = maskvalid
    if topk > numk : topk = numk

    cls_topk = torch.multinomial(probabilities, topk, replacement=False)
    cls_topk = torch.multinomial(probabilities, topk, replacement=False)
 
    final_topk = []
    for i in range(topk):
        clab = cls_topk[i]
        icls_mask = vocab.get_icls_mask(assmbond,clab)
        	
        masked_icls_scores = F.log_softmax(icls_scores + icls_mask, dim=-1)
        
        masked_icls_scores = masked_icls_scores[-1, :] / temperature
        icls_filtered_logits = top_p_filtering(masked_icls_scores, top_p=top_p)
        icls_probabilities = F.softmax(icls_filtered_logits, dim=-1)
            
        icls_numk = len(torch.nonzero(icls_probabilities))
        icls_topk = torch.multinomial(icls_probabilities, 1, replacement=False)
        icls_topk = torch.multinomial(icls_probabilities, 1, replacement=False)

        cls_list.append(clab.cpu())
        #icls_list.append(icls_topk[0].cpu())
        icls_list.append(icls_topk[0].cpu())
 
    return cls_list, icls_list


def root_cls_topP_radom(cls_scores, icls_scores, vocab, num):
    #batch_size = len(cls_scores)
    batch_size = num
    cls_list, icls_list = [], []

    cls_scores = F.log_softmax(cls_scores, dim=-1)

    temperature = 1
    top_p = 0.99
    prob = []

    for x in range(num) :
        cls_scores_tem  = cls_scores[x] / temperature
        filtered_logits = top_p_filtering(cls_scores_tem, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        
        #pronum = np.array(probabilities.cpu())
        #numk = int(sum(pronum>0))
#        print('numk',numk)
        numk = len(torch.nonzero(probabilities))
        nn = 20
        if nn  > numk : nn = numk
        #print('cls numk',numk)
        #cls_topk = torch.multinomial(probabilities, nn, replacement=False)
        #cls_topk = torch.multinomial(probabilities, 1, replacement=False)
        #cls_topk = torch.multinomial(probabilities, 1, replacement=False)
        cls_scores_topk, cls_topk = probabilities.topk(numk, dim=-1)  

        cls_topk = list(cls_topk.tolist())

        cls_topk = random.sample(cls_topk, nn)

        #random.shuffle(cls_topk)
        cls_topk = np.array(cls_topk)
        cls_topk = torch.from_numpy(cls_topk).cuda()
        prob.append(cls_topk)


    #topk = int(len(cls_scores[0]) // 4)
    #topk = int(len(cls_scores))
    #topk = 800
    #if batch_size > topk : topk = batch_size

    #cls_scores_topk, cls_topk = cls_scores.topk(topk, dim=-1)     

#    cls_topk = np.array(cls_topk.cpu())


#    for i in range(len(cls_topk)) :
        #for x in range(int(random.randint(1,5))) :
#        random.shuffle(cls_topk[i])

#    cls_topk = torch.from_numpy(cls_topk).cuda()
    
#    final_topk = []
    totalnum = 0
    for i in range(num):
        #for x in range(topk) :
        #for x in prob[i] :
        #    if x.cpu() not in cls_list :
        #        clab = x
        #        break
        #    clab = cls_topk[i][x]
        #    if clab.cpu() not in cls_list : break
        clsflag = 0
        for jj in range(len(prob[i])) :
            clab = prob[i][jj]
        
            icls_mask = vocab.get_mask([clab])
            sample_icls_scores = F.log_softmax(icls_scores[i] + icls_mask, dim=-1)

        #icls_scores_topk, icls_topk = icls_scores.topk(1, dim=-1)


            sample_icls_scores = sample_icls_scores[-1, :] / temperature
            icls_filtered_logits = top_p_filtering(sample_icls_scores, top_p=top_p)
            icls_probabilities = F.softmax(icls_filtered_logits, dim=-1)
            
        #icls_pronum = np.array(icls_probabilities.cpu())
        #icls_numk = int(sum(icls_probabilities>0))
            icls_numk = len(torch.nonzero(icls_probabilities))
            #print('icls numk',icls_numk)
            #icls_topk = torch.multinomial(icls_probabilities, icls_numk, replacement=False)
        #icls_topk = torch.multinomial(icls_probabilities, 1, replacement=False)
        #icls_topk = torch.multinomial(icls_probabilities, 1, replacement=False)
            icls_scores_topk, icls_topk = icls_probabilities.topk(icls_numk, dim=-1)
            iclflag = 0
            for kk in range(icls_numk) :
                if icls_topk[kk].cpu() not in icls_list :
                    icls_list.append(icls_topk[kk].cpu())
                    iclflag = 1
                    break
            if iclflag == 1 :
                clsflag = 1
                break
        #icls_topk = np.array(icls_topk.tolist())
        #random.shuffle(icls_topk)
        #icls_topk = torch.from_numpy(icls_topk).cuda()        

        if clsflag == 0 :
            clab = prob[i][0]
        
            icls_mask = vocab.get_mask([clab])
            sample_icls_scores = F.log_softmax(icls_scores[i] + icls_mask, dim=-1)

        #icls_scores_topk, icls_topk = icls_scores.topk(1, dim=-1)
            sample_icls_scores = sample_icls_scores[-1, :] / temperature
            icls_filtered_logits = top_p_filtering(sample_icls_scores, top_p=top_p)
            icls_probabilities = F.softmax(icls_filtered_logits, dim=-1)
            
        #icls_pronum = np.array(icls_probabilities.cpu())
        #icls_numk = int(sum(icls_probabilities>0))
            icls_numk = len(torch.nonzero(icls_probabilities))
            icls_topk = torch.multinomial(icls_probabilities, 1, replacement=False)        	
            icls_list.append(icls_topk[0].cpu())
            totalnum = totalnum + 1
        cls_list.append(clab.cpu())
    print('root duplicate number :', totalnum)
    return cls_list, icls_list

def root_cls_topP_Probability(cls_scores, icls_scores, vocab, num):
    #batch_size = len(cls_scores)
    batch_size = num
    cls_list, icls_list = [], []

    cls_scores = F.log_softmax(cls_scores, dim=-1)

    temperature = 1
    top_p = 1
    prob = []

    for x in range(num) :
        cls_scores_tem  = cls_scores[x] / temperature
        filtered_logits = top_p_filtering(cls_scores_tem, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        
        numk = len(torch.nonzero(probabilities))
        cls_topk = torch.multinomial(probabilities, 1, replacement=False)
        #cls_topk = torch.multinomial(probabilities, 1, replacement=False)
        prob.append(cls_topk)
       
    for i in range(num):
        clab = prob[i][0]

        icls_mask = vocab.get_mask([clab])
        sample_icls_scores = F.log_softmax(icls_scores[i] + icls_mask, dim=-1)

        sample_icls_scores = sample_icls_scores[-1, :] / temperature
        icls_filtered_logits = top_p_filtering(sample_icls_scores, top_p=top_p)
        icls_probabilities = F.softmax(icls_filtered_logits, dim=-1)
            
        icls_numk = len(torch.nonzero(icls_probabilities))
        icls_topk = torch.multinomial(icls_probabilities, 1, replacement=False)

        icls_list.append(icls_topk[0].cpu())
        cls_list.append(clab.cpu())
        
    return cls_list, icls_list