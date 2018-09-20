#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

from conf import *

import cPickle
sys.setrecursionlimit(1000000)

random.seed(0)

def sample_action(action_probability):
    ac = action_probability
    ac = ac/ac.sum()
    action = numpy.random.choice(numpy.arange(len(ac)),p=ac)
    return action

def choose_action(action_probability):
    ac_list = list(action_probability)
    action = ac_list.index(max(ac_list))
    return action

# basic utils

def load_pickle(fname):
    with open(fname) as f:
        return cPickle.load(f)

def write_pickle(o, fname):
    with open(fname, 'w') as f:
        cPickle.dump(o, f, -1) 

def load_json_lines(fname):
    with open(fname) as f:
        for line in f:
            yield json.loads(line)

def lines_in_file(fname):
    return int(subprocess.check_output(
        ['wc', '-l', fname]).strip().split()[0])

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
def aa():
    return 0
def get_file_name(path,dir_list):    
    if os.path.isfile(path):
        dir_list.append(path)
    elif os.path.isdir(path):
        for item in os.listdir(path):
            itemsrc = os.path.join(path, item)
            get_file_name(itemsrc,dir_list)
    return dir_list

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = min(len(mcp),len(mp))
    for i in range(0, n): 
        mcp[i].data[:] = mp[i].data[:]
def get_bin(d):                                                                
    if d <= 0:
        return 0                                                               
    elif d == 1:                                                               
        return 1                                                               
    elif d == 2:                                                               
        return 2                                                               
    elif d == 3:                                                               
        return 3                                                               
    elif d == 4:                                                               
        return 4                                                               
    elif d <= 7:                                                               
        return 5                                                               
    elif d <= 15:                                                              
        return 6
    elif d <= 31:                                                              
        return 7
    elif d <= 63:                                                              
        return 8
    else:
        return 9
