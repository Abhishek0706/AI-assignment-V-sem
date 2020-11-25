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
norm = 0
reading_train = True

A = {}  # count of (tag1->tag2)
B = {}  # count of tag,word
tag_count = {}  # count of tag
word_tag_count = defaultdict(dict) # {word : {tag : count}}
Pi = {} # starting probability
tot_Pi = 0

tag_list = []
confusion_matrix = []
index = {}


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
    val_to_ret = []
    if len(list(root)):
        for child in root:
            val_to_ret.extend(parseWord(child))
    else:
        try:
            val_to_ret.append(
                [root.text.strip(), root.attrib.get('c5').split("-")[0]])
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
        if (t, w) in B:
            B[t, w] += 1
        else:
            B[t, w] = 1

        if t in tag_count:
            tag_count[t] += 1
        else:
            tag_count[t] = 1

        # associate tag with word
        if w in word_tag_count:
            if t in word_tag_count[w]:
                word_tag_count[w][t] += 1
            else:
                word_tag_count[w][t] = 1
        else:
            word_tag_count[w][t] = 1


# prob og tag2 given prev tag = tag 1
def trans_prob(tag1, tag2):
    if (tag1, tag2) in A:
        return (1.0 * A[tag1, tag2]) / (1.0 * tag_count[tag1])
    return 0.0


def emmis_prob(tag, word):
    # new word hai
    if word not in word_tag_count:
        return 1.0

    if (tag, word) in B:
        return (1.0 * B[tag, word]) / (1.0 * tag_count[tag])
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

    for i in range(0, n):
        tags = get_tags(word[i])
        for tag in tags:
            dp[i, tag] = 0.0  # initialising dp
            track[i, tag] = "NN1"

    for i in range(0, n):
        tags = get_tags(word[i])
        if i == 0:
            for tag in tags:
                p = start_prob(tag) * emmis_prob(tag, word[i])
                dp[i, tag] = p
        else:
            for tag in tags:
                for prev_tag in prev:
                    p = dp[i-1, prev_tag] * \
                        trans_prob(prev_tag, tag) * emmis_prob(tag, word[i])
                    if p >= dp[i, tag]:
                        dp[i, tag] = p
                        track[i, tag] = prev_tag

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

    # ! predicted tag calc[i] hai lakin jo real tag hai test file me vo multiple
    # ! ho sakte hai to unme se kisme se confusion matrix me add karna hai
    # ! confusion_matrix[predicted][actual] 

    for i in range(0, n):
        if calc[i] in list(t_tag[i].split("-")):
            correct += 1
            confusion_matrix[index[calc[i]]][index[calc[i]]] += 1
        else:
            confusion_matrix[index[calc[i]]][index[t_tag[i].split("-")[0]]] += 1


def get_tags(word):
    val_to_ret = []
    if word in word_tag_count:
        for tag in word_tag_count[word]:
            val_to_ret.append(tag)

    if len(val_to_ret) == 0:
        val_to_ret.extend(tag_list)

    return val_to_ret


def calculate_accuracy():
    global norm
    print("\n------------------")
    accuracy = (100.0 * correct) / (1.0 * total_test)
    print("accuracy: ", "{:.2f}".format(accuracy), "%")


    f1_avg = 0.0
    f1_wt = 0.0
    n = len(confusion_matrix)
    for i in range(n):
        c = confusion_matrix[i][i]
        pre = 0
        rec = 0
        for j in range(n):
            pre += confusion_matrix[i][j]
            rec += confusion_matrix[j][i]
            norm = max(norm,confusion_matrix[i][j])
        precision = (1.0*c)/(1.0 *pre)
        recall = (1.0*c)/(1.0 *rec)
        f1 = (2 * precision * recall)/(precision + recall)
        f1_avg += f1
        f1_wt += rec * f1

    f1_avg = 100 * f1_avg/n
    print("Average F1 :", "{:.2f}".format(f1_avg), "%")

    f1_wt = 100 * f1_wt/total_test
    print("Weighted F1 :", "{:.2f}".format(f1_wt), "%")

    print("------------------\n")


def plot_confusion_matrix():
    row = []
    col = []
    for i in index:
        row.append(i)
        col.append(i)

    normalised = []

    for i in confusion_matrix:
        temp = []
        for j in i:
            temp.append(1.0*j/norm)
        normalised.append(temp)

    df_cm = pd.DataFrame(normalised, index=[i for i in row], columns=[j for j in col])

    plt.figure(figsize=(15, 10.5))
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm)
    plt.title('Confusion matrix')
    plt.savefig("viterbi_plot.png")


# ---------------------------- MAIN METHOD------------------------------------

#  ------------------------train our model ----------------------------------
readFile(trainFolder)

reading_train = False

# -------------------------creating tag list ---------------------------------
for t in tag_count:
    tag_list.append(t)

# -------------------------initialise confusion matrix-----------------------
for _ in range(len(tag_list)):
    temp = []
    for __ in range(len(tag_list)):
        temp.append(0)
    confusion_matrix.append(temp)

random.shuffle(tag_list)
ii = 0
for tt in tag_list:
    index[tt] = ii
    ii += 1

# ---------------------------test our model----------------------------------

readFile(testFolder)

# ---------------------------calculating accuracy----------------------------

calculate_accuracy()

# --------------------------plot confusion matrix--------------------------

plot_confusion_matrix() #also calculation norm with this

# -----------------------------calculating time--------------------------------

print("%s seconds " % (time.time() - start_time))

# --------------------------------THE END-----------------------------------
