import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import TextIO
import seaborn as sn
import pandas as pd
import numpy as np
import random
import csv
import os
import io

import time
start_time = time.time()

trainFolder = './Train-corups'
testFolder = './Test-corpus'

correct = 0
total_test = 0
reading_train = True

A = {} #count of (tag1->tag2)
B = {} # count of tag,word
tag_count = {} # count of tag
word_tag_count = defaultdict(dict)
Pi = {}
tot_Pi = 0

tag_list = []
word_cnt = 0


def readFile(path):
    for fname in os.listdir(path):
        if os.path.isdir(os.path.join(path, fname)):
            print("reading ", os.path.join(path, fname))
            readFile(os.path.join(path, fname))
        else:
            parseFile(os.path.join(path, fname))


def parseFile(path):

    tree = ET.parse(path)
    rootTree = tree.getroot()

    for s_tag in rootTree.iter('s'):
        parseSentence(s_tag)


def parseSentence(root):

    global reading_train
    my_sent = []
    for word in root:
        my_sent.extend(parseWord(word))
    # now we get a sentenceence of train/test
    if reading_train:
        viterbi_train(my_sent)
    else:
        viterbi_test(my_sent)


def parseWord(root):
    global word_cnt
    val_to_ret = []
    if len(list(root)):
        for child in root:
            val_to_ret.extend(parseWord(child))
    else:
        try:
            val_to_ret.append([root.text.strip(),root.attrib.get('c5').split("-")[0]])
            word_cnt += 1
        except:
            None
    return val_to_ret

def viterbi_train(sentence):

    global tot_Pi
    n = len(sentence)
    for i in range(0, n):
        w = sentence[i][0]
        t = sentence[i][1]

        # tranning transition prob
        if i == 0:
            # first word of sentence
            tot_Pi += 1
            if t in Pi:
                Pi[t] += 1
            else:
                Pi[t] = 1
        else:
            prev_t = sentence[i-1][1]
            if (prev_t, t) in A:
                A[prev_t, t] += 1
            else:
                A[prev_t, t] = 1

        # training emmission prob
        if (t,w) in B:
            B[t,w] += 1
        else :
            B[t,w] = 1
        
        if t in tag_count:
            tag_count[t] +=1 
        else :
            tag_count[t] = 1

        # associate tag with word
        if w in word_tag_count:
            if t in word_tag_count[w]:
                word_tag_count[w][t] += 1
            else :
                word_tag_count[w][t] = 1
        else :
            word_tag_count[w][t] = 1


# prob og tag2 given prev tag = tag 1
def trans_prob(tag1, tag2):
    if (tag1,tag2) in A:
        return (1.0 * A[tag1, tag2]) / (1.0 * tag_count[tag1])
    return 0.0


def emmis_prob(tag,word):
    if (tag,word) in B :
        return (1.0 * B[tag,word]) / (1.0 * tag_count[tag])
    return 0.0


def start_prob(tag):
    global tot_Pi
    if tag in Pi:
        return (1.0*Pi[tag])/(1.0 * tot_Pi)
    return 0.0


def viterbi_test(sentence):
    
    word = []
    t_tag = []
    for i in sentence:
        word.append(i[0])
        t_tag.append(i[1])

    n = len(word)

    dp = {}
    track = {}
    prev = [] 

    for i in range(0,n):
        tags = get_tags(word[i])
        for tag in tags:
            dp[i,tag] = 0.0 #initialising dp
            track[i,tag] = "NN1"

    for i in range(0,n):
        tags = get_tags(word[i])
        if i == 0:
            for tag in tags:
                p = start_prob(tag) * emmis_prob(tag,word[i])
                dp[i,tag] = p
        else:
            for tag in tags:
                for prev_tag in prev:
                    p = dp[i-1, prev_tag] * trans_prob(prev_tag,tag) * emmis_prob(tag,word[i])
                    if p >= dp[i,tag]:
                        dp[i,tag]  = p
                        track[i,tag] = prev_tag

        prev = tags

    calc = []
    max_prob = 0.0
    max_tag = ""
    prev_tag = ""


    if n == 1:
        for tag in get_tags(word[n-1]):
            if dp[i, tag] >= max_prob:
                max_prob = dp[i, tag]
                max_tag = tag

        calc.append(max_tag)

    else:
        tags = get_tags(word[n-1])
        for tag in tags:
            if(dp[n-1, tag] >= max_prob):
                max_prob = dp[n-1, tag]
                max_tag = tag
                prev_tag = track[n-1, tag]

        calc.append(max_tag)
        for i in range(n-2, 0, -1):
            max_tag = prev_tag
            calc.append(max_tag)
            prev_tag = track[i, max_tag]
        calc.append(prev_tag)

        calc.reverse()

    global correct
    global total_test

    total_test += n

    for i in range(0,n):
        if calc[i] in list(t_tag[i].split("-")):
            correct += 1


def get_tags(word):
    val_to_ret = []
    if word in word_tag_count:
        for tag in word_tag_count[word]:
            val_to_ret.append(tag)

    if len(val_to_ret) == 0:
        val_to_ret.append("NN1")
        # TODO : iska kuch karna pdega.....kuki us stage ki saari prob 0 ho rhi hai

    return val_to_ret


# ---------------------------- MAIN METHOD------------------------------------

#  ------------------------train our model ----------------------------------
readFile(trainFolder)

reading_train = False

# -------------------------creating tag list ---------------------------------
for t in tag_count:
    tag_list.append(t)

# ---------------------------test our model----------------------------------
readFile(testFolder)

accuracy = (1.0 * correct) / (1.0 * total_test)

print(accuracy)

# -----------------------------------------------------------------------------

print("--- %s seconds ---" % (time.time() - start_time))
