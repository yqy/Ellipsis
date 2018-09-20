#coding=utf8
import sys
import os
import re
import parse_analysis
import collections
import copy

class Doc:
    def __init__(self):
        self.zps = {}
        self.azps = {}
        self.nps = {}
        self.filter_nodes = {}
        self.nodes = {}
        self.index2real = {}
        self.zp_dict = {}
        self.np_dict = {}
        self.sentence_num = 0
        self.vec = [] #vectorize sentence
        self.mask = []
        self.zp_pre_index = []
        self.zp_post_index = []
        self.sentence_len = []
        self.np_index_start = []
        self.np_index_end = []
        self.np_indecs = []
        self.np_mask = []
        self.zp2candi_dict = {}
        self.zp_candi_distance_dict = {}
        self.np2real = []
        self.zp2real = []
        self.zp_candi_coref = {}
    def init_sentence(self,sen_id):
        self.sentence_num += 1
        self.zps[sen_id] = []
        self.azps[sen_id] = []
        self.nps[sen_id] = []
        self.filter_nodes[sen_id] = []
        self.nodes[sen_id] = []
        self.index2real[sen_id] = {}
    def add_zp(self,sen_id,index):
        new_zp = ZP(sen_id=sen_id,index=index)
        self.zps[sen_id].append(new_zp)
        self.zp_dict[(sen_id,index)] = self.zps[sen_id][-1]

    def add_np(self,sen_id,start_index,end_index):
        new_np = NP(start=start_index,end=end_index,sen_id=sen_id)
        self.nps[sen_id].append(new_np)
        self.np_dict[(sen_id,start_index,end_index)] = self.nps[sen_id][-1]

    def update(self):
        self.all_zps = []
        self.all_azps = []
        for si in self.zps:
            for zp in self.zps[si]:
                self.all_zps.append(zp)
                if zp.azp:
                    self.azps[si].append(zp)
                    self.all_azps.append(zp)


class NP:
    def __init__(self, start=-1, end=-1, coref_id=-1, sen_id=-1):
        self.start = start
        self.end = end
        self.coref_id = coref_id
        self.sen_id = sen_id
class ZP:
    def __init__(self, azp=False, index=-1, coref_id=-1, sen_id=-1, candidate=[], antecedent=[]):
        self.azp = azp
        self.index = index
        self.coref_id = coref_id
        self.sen_id = sen_id
        self.candidate = candidate
        self.antecedent = []
    def set_azp(self,coref_id):
        self.coref_id = coref_id
        self.azp = True
    def set_unazp(self):
        self.azp = False
    def set_antecedent(self,ante):
        self.antecedent = copy.copy(ante)
