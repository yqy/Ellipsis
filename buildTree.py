#coding=utf8
import os
import sys
import re
import parse_analysis
from subprocess import *
import document
from conf import *

def is_zero_tag(leaf_nodes):
    if len(leaf_nodes) == 1:
        if leaf_nodes[0].word.find("*") >= 0:
            return True
    return False

def is_np(tag):
    np_list = ['NP-SBJ', 'NP', 'NP-PN-OBJ', 'NP-PN', 'NP-PN-SBJ', 'NP-OBJ', 'NP-TPC-1', 'NP-TPC', 'NP-PN-VOC', 'NP-VOC', 'NP-IO', 'NP-SBJ-1', 'NP-PN-TPC', 'NP-PRD', 'NP-TMP', 'NP-PN-PRD', 'NP-PN-SBJ-1', 'NP-APP', 'NP-TPC-2', 'NP-PN-SBJ-3', 'NP-PN-IO', 'NP-PN-LOC', 'NP-SBJ-2', 'NP-PN-OBJ-1', 'NP-LGS', 'NP-MNR', 'NP-SBJ-3', 'NP-OBJ-PN', 'NP-SBJ-4', 'NP-PN-SBJ-2', 'NP-TPC-3', 'NP-HLN', 'NP-PN-APP', 'NP-SBJ-PN', 'NP-DIR', 'NP-LOC', 'NP-ADV', 'NP-WH-SBJ']
    if tag in np_list:
        return True
    else:
        return False

def get_info_from_file(file_name,MAX=2):

    pattern = re.compile("(\d+?)\ +(.+?)$")
    pattern_zp = re.compile("(\d+?)\.(\d+?)\-(\d+?)\ +(.+?)$")

    inline = "new"
    f = open(file_name)

    doc = document.Doc() 
    sentence_num = 0

    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()

        if line == "Leaves:":
            while True:
                inline = f.readline()
                if inline.strip() == "":break
                inline = inline.strip()
                match = pattern.match(inline)
                if match:
                    word = match.groups()[1]
            sentence_num += 1
    
        elif line == "Tree:":
            doc.init_sentence(sentence_num)
            parse_info = ""
            inline = f.readline()
            while True:
                inline = f.readline()
                if inline.strip("\n") == "":break
                parse_info = parse_info + " " + inline.strip()    
            parse_info = parse_info.strip()            
            nl,wl = parse_analysis.buildTree(parse_info)

            index_without_null = 0
            all_words_wl = []
            for node in wl:
                if node.word.find("*") < 0: #not a pro
                    new_node = parse_analysis.Node()
                    new_node.copy_from(node)
                    new_node.index = index_without_null
                    doc.index2real[sentence_num][node.index] = new_node.index
                    index_without_null += 1
                    all_words_wl.append(new_node)
                else:
                    doc.index2real[sentence_num][node.index] = index_without_null

                if node.word == "*pro*":
                    doc.add_zp(sentence_num,doc.index2real[sentence_num][node.index])
            doc.nodes[sentence_num] = wl
            doc.filter_nodes[sentence_num] = all_words_wl

            for node in nl:
                if is_np(node.tag):
                    if node.parent.tag.startswith("NP"):
                        if not (node == node.parent.child[0]):
                            continue
                    leaf_nodes = node.get_leaf()
                    if is_zero_tag(leaf_nodes):
                        continue
                    doc.add_np(sentence_num,doc.index2real[sentence_num][leaf_nodes[0].index],doc.index2real[sentence_num][leaf_nodes[-1].index]) 
        elif line.startswith("Coreference chain"):
            first = True
            res_info = None
            last_index = 0
            antecedents = []

            while True:
                inline = f.readline()
                if not inline:break
                if inline.startswith("----------------------------------------------------------------------------------"):
                    break
                inline = inline.strip()
                if len(inline) <= 0:continue
                if inline.startswith("Chain"):
                    first = True
                    res_info = None
                    last_index = 0
                    antecedents = []
                    coref_id = inline.strip().split(" ")[1]
                else:
                    match = pattern_zp.match(inline)
                    if match:
                        sentence_index = int(match.groups()[0])
                        begin_word_index = int(match.groups()[1])
                        end_word_index = int(match.groups()[2])
                        word = match.groups()[-1]
                        if word == "*pro*":
                            is_azp = False
                            if not first:
                                is_azp = True
                                if doc.zp_dict.has_key((sentence_index,doc.index2real[sentence_index][begin_word_index])):
                                    this_zp = doc.zp_dict[(sentence_index,doc.index2real[sentence_index][begin_word_index])]
                                    this_zp.set_azp(coref_id)
                                    this_zp.set_antecedent(antecedents)
                        if not word == "*pro*":
                            first = False
                            res_info = inline
                            last_index = sentence_index
                            if doc.np_dict.has_key(((sentence_index,doc.index2real[sentence_index][begin_word_index],doc.index2real[sentence_index][end_word_index]))):
                                this_np = doc.np_dict[(sentence_index,doc.index2real[sentence_index][begin_word_index],doc.index2real[sentence_index][end_word_index])]
                                this_np.coref_id = coref_id
                                antecedents.append(this_np)
        
        if not inline:
            break
    doc.update()
    return doc

def main(files):
    azp_num = 0
    zp_num = 0
    sen_num = 0
    doc_num = 0
    paths = [w.strip() for w in open(files).readlines()]
    for p in paths:
        if p.strip().endswith("DS_Store"):continue
        doc_num += 1
        file_name = p.strip()
        doc = get_info_from_file(file_name,2)
        azp_num += len(doc.all_azps)
        zp_num += len(doc.all_zps)
        sen_num += doc.sentence_num
    print doc_num,sen_num,zp_num,azp_num

if __name__ == "__main__":
    main("./data/test_list")
    #main("./data/train_list")
