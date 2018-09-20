#coding=utf8
import os
import sys
import re
import math
import timeit
from subprocess import *
from conf import *
import cPickle
import collections
sys.setrecursionlimit(1000000)

class DataGnerater():
    def __init__(self,file_type,devide=False,k=0.2):
        data_path = args.data+file_type+"/" 
        if args.reduced == 1:
            data_path = args.data+file_type + "_reduced/"
        read_f = file(data_path + "docs","rb")
        docs = cPickle.load(read_f)
        read_f.close()
        self.data_batch = docs
        if devide:
            self.devide(k=k)
        
    def devide(self,k=0.2):
        random.shuffle(self.data_batch)
        length = int(len(self.data_batch)*k)
        self.dev = self.data_batch[:length]
        self.train = self.data_batch[length:]
        self.data_batch = self.train

    def generate_data(self,shuffle=False,dev=False):
        if shuffle:
            random.shuffle(self.data_batch) 
        estimate_time = 0.0 
        done_num = 0 
        if dev:
            total_num = len(self.dev)
            for data in self.dev:
                start_time = timeit.default_timer()
                done_num += 1
                yield data
                end_time = timeit.default_timer()
                estimate_time += (end_time-start_time)
                EST = total_num*estimate_time/float(done_num)
                info = "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)
                sys.stderr.write(info+"\r")
            print >> sys.stderr
        else:
            total_num = len(self.data_batch)
            for data in self.data_batch:
                start_time = timeit.default_timer()
                done_num += 1
                yield data
                end_time = timeit.default_timer()
                estimate_time += (end_time-start_time)
                EST = total_num*estimate_time/float(done_num)
                info = "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)
                sys.stderr.write(info+"\r")
            print >> sys.stderr
