#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
from subprocess import *

from conf import *
from buildTree import get_info_from_file
import utils
import opt
from data_generater import *
random.seed(0)
numpy.random.seed(0)

import cPickle
sys.setrecursionlimit(1000000)

MAX = 2

def setup():
    utils.mkdir(args.data)
    utils.mkdir(args.data+"train/")
    utils.mkdir(args.data+"train_reduced/")
    utils.mkdir(args.data+"test/")
    utils.mkdir(args.data+"test_reduced/")

def list_vectorize(wl,words):
    il = []
    for w in wl:
        word = w.word
        if word in words:
            index = words.index(word)
        else:
            index = 0
        il.append(index) 
    return il
def mask_array(embedding_vec):
    max_length = max([len(em) for em in embedding_vec])
    out_em = []
    out_mask = []
    for em in embedding_vec:
        out_em.append(em+[0]*(max_length-len(em)))
        out_mask.append([1]*len(em)+[0]*(max_length-len(em)))
    return numpy.array(out_em),numpy.array(out_mask)

def generate_doc_data(path,files):
    paths = [w.strip() for w in open(files).readlines()]
    docs = []
    done_num = 0
    for p in paths:
        if p.strip().endswith("DS_Store"):continue
        done_num += 1
        file_name = p.strip()
        if file_name.endswith("onf"):
            if args.reduced == 1 and done_num >= 15:break
            doc = get_info_from_file(file_name,2)
            docs.append(doc) 
    return docs

def generate_data(file_name="",test_only=False):

    startt = timeit.default_timer()
    DATA = args.raw_data

    train_data_path = args.data + "train/"+file_name
    test_data_path = args.data + "test/"+file_name
    if args.reduced == 1:
        train_data_path = args.data + "train_reduced/"+file_name
        test_data_path = args.data + "test_reduced/"+file_name

    docs = generate_doc_data(DATA+"test/"+file_name,"./data/test_list")
    print "go"
    generate_vec(test_data_path,docs)

    endt = timeit.default_timer()
    print >> sys.stderr
    print >> sys.stderr, "Total use %.3f seconds for Data Generating"%(endt-startt)


def generate_vec(data_path,docs):
    read_f = file("./data/emb","rb")
    embedding,words,_ = cPickle.load(read_f)
    read_f.close()
    #np_len = [5,6,7,8,9,10]
    np_len = [3,4,5]
    for doc in docs:
        coref_nps = []
        for ZP in doc.all_azps:
            for np in ZP.antecedent:
                coref_nps.append(np)
        doc.coref_nps = coref_nps

    for width in np_len:
        nps = 0
        hit_np = 0
        hit_ante = 0
        hit_azp = 0
        for doc in docs:
            # generate vector for this doc
            vectorized_sentences = []
            for i in range(doc.sentence_num):
                nodes = doc.filter_nodes[i]
                vectorize_words = list_vectorize(nodes,words) 
                vectorized_sentences.append(vectorize_words)
            vec,mask = mask_array(vectorized_sentences)
            doc.vec = vec
            doc.mask = mask
            inpt_len = numpy.sum(doc.mask,1)
            doc.sentence_len = inpt_len
            pre_index,post_index,zp2sen_dict,zp2real = opt.generate_ZP_index(inpt_len)
            doc.zp_pre_index = pre_index
            doc.zp_post_index = post_index
            np_index_start,np_index_end,np_indecs,np_mask,np_sen2index_dict,np2real = opt.generate_NP_index(inpt_len,MAX=width)
            nps += len(np_indecs)

            #doc.np_dict[(this_np_real_sentence_num,this_np_real_start,this_np_real_end)]
            np_d = {}
            gold_np = [0]*len(np2real)
            for i,(this_np_real_sentence_num,this_np_real_start,this_np_real_end) in enumerate(np2real):
                if (this_np_real_sentence_num,this_np_real_start,this_np_real_end) in doc.np_dict:
                    gold_np[i] = 1
                    hit_np += 1 
                    this_np = doc.np_dict[(this_np_real_sentence_num,this_np_real_start,this_np_real_end)]
                    if this_np in doc.coref_nps:
                        hit_ante += 1
                    np_d[this_np] = (this_np_real_sentence_num,this_np_real_start,this_np_real_end)

            for ZP in doc.all_azps:
                for np in ZP.antecedent:
                    if np in doc.coref_nps:
                        if np in np_d:
                            this_np_real_sentence_num,this_np_real_start,this_np_real_end = np_d[np]
                            if (this_np_real_sentence_num >= (ZP.sen_id-2) and this_np_real_sentence_num < ZP.sen_id) or (this_np_real_sentence_num == ZP.sen_id and this_np_real_end <= ZP.index):
                                hit_azp += 1
                                break
            continue
        
        print width,"All",nps,"NPs",hit_np,"Antes",hit_ante,float(hit_np)/float(nps),float(hit_ante)/float(nps),hit_azp
    return nps

    if 1:
        doc.np_index_start = np_index_start
        doc.np_index_end = np_index_end
        doc.np_indecs = np_indecs
        doc.np_mask = np_mask
        doc.zp2real = zp2real # zp_index: real_sentence_num,real_index
        doc.np2real = np2real # np_index: real_sentence_num,real_start_index,real_end_index
        
        zp2candi_dict = {} #zp_index: [np_index]
        for i in range(len(pre_index)):
            zp_index = pre_index[i]
            sen_index = zp2sen_dict[i]
            zp2candi_dict[i] = []
            for sen_id in range(max(0,sen_index-2),sen_index+1):
                np_indexs = np_sen2index_dict[sen_id] 
                for np_index in np_indexs:
                    if not ( (sen_id == sen_index) and (np_index_end[np_index] > zp_index) ):
                        zp2candi_dict[i].append(np_index)
        doc.zp2candi_dict = zp2candi_dict
        zp_candi_distance_dict = {}
        zp_candi_coref = {} # zp_index: [0,1,0,0,0] coreference result for each of its candidate
        gold_azp = []
        gold_np = [0]*len(np2real)
        for zp_index in doc.zp2candi_dict:
            gold_azp_add = 0
            if len(doc.zp2candi_dict[zp_index]) > 0:
                this_zp_real_sentence_num,this_zp_real_index = doc.zp2real[zp_index]
                this_zp = None
                if (this_zp_real_sentence_num,this_zp_real_index) in doc.zp_dict:
                    this_zp = doc.zp_dict[(this_zp_real_sentence_num,this_zp_real_index)]

                np_indexes = doc.zp2candi_dict[zp_index]
                max_index = max(np_indexes)
                #zp_candi_coref[zp_index] = numpy.array([0]*(max(np_indexes)+2)) #last index = 1 means zp is not azp
                zp_candi_coref[zp_index] = numpy.array([0]*(len(np_indexes)+1)) #last index = 1 means zp is not azp
                zp_candi_distance_dict[zp_index] = [] #utils.get_bin(distance)
                for ii, np_index in enumerate(np_indexes):
                    distance = max_index-np_index
                    zp_candi_distance_dict[zp_index].append(utils.get_bin(distance))

                    this_np_real_sentence_num,this_np_real_start,this_np_real_end = doc.np2real[np_index]
                    if (this_np_real_sentence_num,this_np_real_start,this_np_real_end) in doc.np_dict:
                        this_candi = doc.np_dict[(this_np_real_sentence_num,this_np_real_start,this_np_real_end)]
                        if this_zp:
                            if this_candi in this_zp.antecedent:
                                zp_candi_coref[zp_index][ii] = 1
                                gold_np[np_index] = 1
                gold_azp_add = 1
                if sum(zp_candi_coref[zp_index]) == 0:
                    zp_candi_coref[zp_index][-1] = 1
                    gold_azp_add = 0
            gold_azp.append(gold_azp_add)
        
        doc.zp_candi_coref = zp_candi_coref
        doc.zp_candi_distance_dict = zp_candi_distance_dict
        doc.gold_azp = gold_azp
        doc.gold_np = gold_np

    save_f = file(data_path + "docs", 'wb')
    cPickle.dump(docs, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()
  
if __name__ == "__main__":
    # build data from raw OntoNotes data
    setup()
    generate_data()
