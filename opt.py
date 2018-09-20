import numpy as np
from conf import *

def generate_ZP_index(inpt_len):
    pre_index = []
    post_index = []
    zp2sen = {}
    base = 0 
    sen_index = 0
    zp_index = 0
    zp2real = []
    for length in inpt_len:
        pre_index.append(-1) 
        zp2sen[zp_index] = sen_index
        zp_index += 1
        for i in range(length):
            pre_index.append(i+base) 
            post_index.append(i+base)
            zp2real.append((sen_index,i))
            zp2sen[zp_index] = sen_index
            zp_index += 1
        post_index.append(-1) 
        zp2real.append((sen_index,-1))
        base += length 
        sen_index += 1
    return pre_index,post_index,zp2sen,zp2real

def generate_NP_index(inpt_len,MAX=10):
    start_index = []
    end_index = []
    sen2index = {}
    mask = []
    base = 0 
    sen_index = 0
    np2real = []
    sen2index[sen_index] = []
    np_index = 0
    for length in inpt_len:
        for i in range(length):
            for j in range(i,min(length,i+MAX)):
                start_index.append(i+base) 
                end_index.append(j+base)
                mask.append([1]*(j-i+1)+[0]*(MAX-j+i-1))
                sen2index[sen_index].append(np_index)
                np2real.append((sen_index,i,j))
                np_index += 1
        base += length 
        sen_index += 1
        sen2index[sen_index] = []
    starts = np.array(start_index)
    ends = np.expand_dims(np.array(end_index),1)
    mention_indices = np.expand_dims(starts, axis=1)+ np.expand_dims(np.array(range(MAX)),axis=0)
    mention_indices = np.minimum(mention_indices,ends)
    #mention_indices = np.minimum(mention_indices,len()-1)
    return np.array(start_index),np.array(end_index),mention_indices,np.array(mask),sen2index,np2real

def get_candis_by_scores(gold_np,starts,ends,scores,lamda=0.3,train=False):
    num_output_mentions = min(int(len(starts)*lamda)+1,len(starts))
    sorted_index = numpy.argsort(scores)[::-1]
    top_mention_indices = []
    current_mention_index = 0 
    while (len(top_mention_indices) < num_output_mentions):
        i = sorted_index[current_mention_index]
        any_crossing = False
        for j in top_mention_indices:
            if is_crossing(starts,ends,i,j):
                any_crossing = True
                break
        if not any_crossing:
            top_mention_indices.append(i)
        current_mention_index += 1
    if train:
        for i in gold_np:
            if not i in top_mention_indices:
                top_mention_indices = numpy.append(top_mention_indices,i)
                #top_mention_indices.append(i) 

    top_mention_indices.sort()
    return top_mention_indices

def is_crossing(starts,ends,i,j):
    s1 = starts[i]
    s2 = starts[j]
    e1 = starts[i]
    e2 = starts[j]
    return (s1 < s2 and s2 <= e1 and e1 < e2) or (s2 < s1 and s1 <= e2 and e2 < e1) 

def get_zps_by_scores(gold,scores,lamda=0.1,train=False):
    num_output_mentions = min(int(len(scores)*lamda)+1,len(scores))
    sorted_index = numpy.argsort(scores)[::-1]
    top_mention_indices = sorted_index[:num_output_mentions]

    if train:
        for i in gold:
            if not i in top_mention_indices:
                top_mention_indices = numpy.append(top_mention_indices,i) 

    top_mention_indices.sort()

    return top_mention_indices.copy()

def get_pairs(selected_zp_indexs,selected_np_indexs,doc):
    zp_reindex = []
    np_reindex = []
    np_reindex_real = []
    distance_feature = []
    start = []
    end = []
    smallest = 0
    for i,zp_index in enumerate(selected_zp_indexs):
        candi_indexes = doc.zp2candi_dict[zp_index]
        s = len(zp_reindex)
        if len(candi_indexes) > 0:
            max_candi_index = max(candi_indexes)
            this_small = 1000000
            for j in range(smallest,len(selected_np_indexs)):
                candi_index = selected_np_indexs[j]
                if candi_index > max_candi_index:
                    break
                if candi_index in candi_indexes:
                    this_candi_index_in_candi_list = candi_indexes.index(candi_index)
                    zp_reindex.append(zp_index)
                    np_reindex.append(this_candi_index_in_candi_list) 
                    np_reindex_real.append(candi_index) 
                    distance_feature.append(doc.zp_candi_distance_dict[zp_index][this_candi_index_in_candi_list])
                    #if candi_index < this_small:
                    if j < this_small:
                        this_small = j

            if this_small < 1000000:
                smallest = this_small
        e = len(zp_reindex)
        if not s == e:
            start.append(s)
            end.append(e)
    return zp_reindex,np_reindex,np_reindex_real,start,end,distance_feature
