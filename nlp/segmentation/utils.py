#!/usr/bin/env python
# -*- coding: utf-8 -*-


import codecs
from functools import reduce
import os
import numpy as np

from ..utils import *

def load_txt(x):
    with open(x, 'rb') as f:
        res = [t.decode('gbk', 'ignore') for t in f]
        return ''.join(res)
# or use codecs.open
    # with codecs.open(x, encoding='gbk') as f:
        # doc = f.read()
        # return doc

def load_sogou_reduced(path='../../../data/text/Reduced'):
    dirs = os.listdir(path)
    dirs = [os.path.join(path, f) for f in dirs if f.startswith('C')]

    text_t = {}
    for i, d in enumerate(dirs):
        files = os.listdir(d)
        files = [os.path.join(d, x) for x in files if x.endswith(
            'txt') and not x.startswith('.')]
        text_t[i] = [load_txt(f) for f in files]

    return text_t

#################### tagging ###########################################
def tag_character_BMES(input_file, output_file):
    """
        tag character with B, M, E, S from segmented sentences
        B: beginning of a word
        M: middle of a word
        E: end of a word
        S: single character as a word
    """
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "/S ")
            else:
                output_data.write(word[0] + "/B ")
                for w in word[1:len(word) - 1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word) - 1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()

def untag_character_BMES(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    # 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
    for line in input_data.readlines():
        char_tag_list = line.strip().split()
        for char_tag in char_tag_list:
            char_tag_pair = char_tag.split('/')
            char = char_tag_pair[0]
            tag = char_tag_pair[1]
            if tag == 'B':
                output_data.write(' ' + char)
            elif tag == 'M':
                output_data.write(char)
            elif tag == 'E':
                output_data.write(char + ' ')
            else:  # tag == 'S'
                output_data.write(' ' + char + ' ')
        output_data.write("\n")
    input_data.close()
    output_data.close()

def predict_tags(input_windows, input_sentence,
                 model,
                 label_to_num,
                 num_to_label=None):
    input_windows = np.array(input_windows)
    predict_prob = model.predict_proba(input_windows)
    predict_label = predict_prob.argmax(-1)
    print(input_windows.shape,
          reduce(lambda prev, x: prev + len(x.strip()),
                 input_sentence.split('\n'), 0),
          )
    # predict_label = model.predict_classes(input_windows)
    for i, lable in enumerate(predict_label[:-1]):
        # 如果是首字 ，不可为E, M
        if i == 0:
            predict_prob[i, label_to_num[u'E']] = 0
            predict_prob[i, label_to_num[u'M']] = 0
        # 前字为B，后字不可为B,S
        if lable == label_to_num[u'B']:
            predict_prob[i + 1, label_to_num[u'B']] = 0
            predict_prob[i + 1, label_to_num[u'S']] = 0
        # 前字为E，后字不可为M,E
        if lable == label_to_num[u'E']:
            predict_prob[i + 1, label_to_num[u'M']] = 0
            predict_prob[i + 1, label_to_num[u'E']] = 0
        # 前字为M，后字不可为B,S
        if lable == label_to_num[u'M']:
            predict_prob[i + 1, label_to_num[u'B']] = 0
            predict_prob[i + 1, label_to_num[u'S']] = 0
        # 前字为S，后字不可为M,E
        if lable == label_to_num[u'S']:
            predict_prob[i + 1, label_to_num[u'M']] = 0
            predict_prob[i + 1, label_to_num[u'E']] = 0
        # viterbi algorithm
        predict_label[i + 1] = predict_prob[i + 1].argmax()
    tags = [num_to_label[x] for x in predict_label]

    # tag each word
    j = 0
    result = ''
    for line in input_sentence.split('\n'):
        for i, c in enumerate(line.strip()):
            result = '%s %s/%s' % (result, c, tags[j])
            j += 1
        result = '%s\n' % (result)
    return result

#################### COMMON UTILS ########################################
