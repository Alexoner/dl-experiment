#!/usr/bin/env python
# -*- coding: utf-8 -*-


from logging import Logger
import itertools
import numpy as np
from keras.callbacks import Callback



def flatten(lst):
    """ flatten [[]] into []
    """
    return list(itertools.chain.from_iterable(lst))

def list_to_mappings(lst):
    num_to_item = dict(zip(range(len(lst)), np.unique(lst)))
    item_to_num = {item: num for num, item in num_to_item.items()}
    return num_to_item, item_to_num

def invert_dict(d):
    return {v: k for k, v in d.items()}

def loadwv(wvfile):
    wv = np.loadtxt(wvfile, dtype=float)
    return wv

# cut character
def trim_space(s):
    """ trim white spaces of a `str` """
    return ''.join(s.split())

def cutchar(x):
    """ separate characters by space """
    words = list(x)
    return ' '.join(words)

def extract_tag_set(docs):
    tags = set(flatten([[t[1].split("/")[0] for t in d] for d in docs]))
    return tags

def extract_word_set(docs):
    words = set(flatten([[t[0] for t in d] for d in docs]))
    return words

def pad_sequence(seq, left=1, right=1, padding=[("<s>", ""), ("</s>", "")]):
    # pad leading and trailing
    return int(left) * [padding[0]] + seq + int(right) * [padding[1]]

def sent_to_seq(sentence, left=1, right=1, seq_func=lambda x: list(x)):
    """ sentence to sequence whiling preserving white spaces
    @args
    sentence: str sentence
    left: left padding number
    right: right padding number
    seq_func: convert a sentence into a sequence. x.split() for English-like languages
        which are separated by white spaces. list(x) for Chinese without white spaces.
    """
    return pad_sequence(seq_func(sentence.strip()),
                        left, right, padding=['<s>', '</s>'])

def doc_to_seq(doc, left=0, right=0):
    """
    @args:
        doc:
        list of sentences

    @return:
        list of sequences of tokens:[[]]
    """
    return list(map(lambda sentence: sent_to_seq(sentence, left, right), doc))
##
# For window models
def seq_to_windows(words, tags, word_to_num, tag_to_num=None, left=1, right=1):
    ns = len(words)
    X = []
    y = []
    for i in range(ns):
        if words[i] == "<s>" or words[i] == "</s>":
            continue  # skip sentence delimiters
        # 文本中的字如果在词典中则转为数字，如果不在则设置为'<UNK>' (UNKNOWN)
        idxs = [word_to_num[words[ii]] if words[ii] in word_to_num else word_to_num['<UNK>']
                for ii in range(i - int(left), i + int(right) + 1)]
        X.append(idxs)
        if tag_to_num is not None:
            tagn = tag_to_num[tags[i]]
            y.append(tagn)
    return np.array(X), np.array(y)

def sent_to_windows(sentence, word_to_num, window=7):
    padding = int((window - 1) * 0.5)
    seq = sent_to_seq(sentence, padding, padding)
    X, _ = seq_to_windows(seq, None, word_to_num, left=padding, right=padding)
    return X

def doc_to_windows(doc, word_to_num, window=7):
    doc_windows = []
    for sent_windows in map(
            lambda x: sent_to_windows(x, word_to_num, window), doc):
        doc_windows.extend(sent_windows)
    return doc_windows

def docs_to_windows(docs, word_to_num, tag_to_num, wsize=3):
    pad = (wsize - 1) / 2
    docs = flatten([pad_sequence(seq, left=pad, right=pad) for seq in docs])

    words, tags = list(zip(*docs))
    # words = [canonicalize_word(w, word_to_num) for w in words]
    tags = [t.split("|")[0] for t in tags]
    return seq_to_windows(words, tags, word_to_num, tag_to_num, pad, pad)

def window_to_vec(window, L):
    """Concatenate word vectors for a given window.
    @args
        L: Look Up Table
    """
    return np.concatenate([L[i] for i in window])

class EarlyStopping(Callback):
    def __init__(self, patience=0, verbose=0):
        super().__init__()

        self.patience = patience
        self.verbose = verbose
        self.best_val_loss = np.Inf
        self.wait = 0
        self.logger = Logger(EarlyStopping.__name__)

    def on_epoch_end(self, epoch, logs={}):
        if not self.params['do_validation']:
            self.logger.warning("Early stopping requires validation data!")

        cur_val_loss = logs.get('val_loss')
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1
