# New words are consider to be in NN1

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
frequency_list_train = []
wordCount_list_train = []
tagCount_list_train = []
word_tag_list_train = []
confusion_matrix = []
viterbi_confusion_matrix = []
word_tag_list_test = []
word_tag_cnt_dict_train = {}
word_cnt_dict_train = {}
tag_cnt_dict_train = {}
predicted_dict = {}
index = {}
A = {}
B = {}
Pi = {}
tot_Pi = 0
viterbi = {}

# we have method named getA(prev, curr) which give the prob of prevtag = prev and currtag = curr i.e Aprev,curr
# we have method named getB(word,tag) which give prob to word given tag
# we have method named getPi(tag) which give prob to tag to start a sentence

correct = 0
total_test = 0
accuracy = 0
norm = 0.0
reading_train = True

viterbi_correct = 0
viterbi_total_test = 0
viterbi_accuracy = 0
viterbi_norm = 0.0



def readFile(path):
    val_to_ret = []
    for fname in os.listdir(path):
        if os.path.isdir(os.path.join(path, fname)):
            print(os.path.join(path, fname))
            val_to_ret.extend(readFile(os.path.join(path, fname)))
        else:
            val_to_ret.extend(parseFile(os.path.join(path, fname)))
    return val_to_ret


def parseFile(path):
    tree = ET.parse(path)
    rootTree = tree.getroot()
    val_to_ret = []

    for sent in rootTree.iter('s'):
        val_to_ret.extend(parseSentence(sent))

    return val_to_ret


def parseSentence(root):
    global reading_train
    val_to_ret = []
    for word in root:
        val_to_ret.extend(parseWord(word))

    # now we get a sentence of train/test
    if reading_train:
        viterbi_train(val_to_ret)
    else:
        viterbi_test(val_to_ret)

    return val_to_ret


def parseWord(root):
    val_to_ret = []
    if len(list(root)):
        for child in root:
            val_to_ret.extend(parseWord(child))

    else:
        try:
            li = list(root.attrib.get('c5').split("-"))
            for c5word in li:
                val_to_ret.append([root.text.strip(), c5word.strip()])
        except:
            None

    return val_to_ret


def create_dict(word_tag_list):
    dict_word = defaultdict(dict)
    dict_cnt_c5 = defaultdict(int)
    dict_cnt_word = defaultdict(int)
    for x in word_tag_list:
        if x[0] in dict_word:
            if x[1] in dict_word[x[0]]:
                dict_word[x[0]][x[1]] += 1
            else:
                dict_word[x[0]][x[1]] = 1
        else:
            dict_word[x[0]][x[1]] = 1
        dict_cnt_c5[x[1]] += 1
        dict_cnt_word[x[0]] += 1
    return [dict_word, dict_cnt_word, dict_cnt_c5]


def cmp(aa):
    return aa[1]


def create_readable_format():
    print("creating readable format file....")
    file: TextIO
    with io.open('./readableFormat.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(word_tag_list_train)


def create_frequency_list():
    print("creating frequency list....")
    for word in word_tag_cnt_dict_train:
        for tag in word_tag_cnt_dict_train[word]:
            frequency_list_train.append(
                [word, tag, word_tag_cnt_dict_train[word][tag]])
    print("creating word tag frequency list....")
    file: TextIO
    with io.open('./frequencyTable.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(frequency_list_train)


def create_word_count_list():
    print("creating word count list....")
    for word in word_cnt_dict_train:
        wordCount_list_train.append([word, word_cnt_dict_train[word]])
    wordCount_list_train.sort(reverse=True, key=cmp)
    print("generating top 10 most used word....")
    file: TextIO
    with io.open('./wordCount.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        for x in range(0, 10):
            writer.writerow(wordCount_list_train[x])


def create_tag_count_list():
    print("creating tag count list....")
    for tag in tag_cnt_dict_train:
        tagCount_list_train.append([tag, tag_cnt_dict_train[tag]])
    tagCount_list_train.sort(reverse=True, key=cmp)
    print("generating top 10 used tag....")
    file: TextIO
    with io.open('./tagCount.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        for x in range(0, 10):
            writer.writerow(tagCount_list_train[x])


def create_predicted_dictionary():
    print("creating predicted dictionary....")

    for word in word_tag_cnt_dict_train:
        maxi = 0
        maxi_tag = ""
        for tag in word_tag_cnt_dict_train[word]:
            if word_tag_cnt_dict_train[word][tag] > maxi:
                maxi = word_tag_cnt_dict_train[word][tag]
                maxi_tag = tag
        predicted_dict[word] = maxi_tag


def create_confusion_matrix():
    global total_test
    global correct
    global norm
    global accuracy
    print("creating confusion matrix....")

    # initialising confusion matrix
    n = len(tagCount_list_train)
    for _ in range(n):
        temp = []
        for __ in range(n):
            temp.append(0)
        confusion_matrix.append(temp)

    # mapping tag to index

    new_word_fount = 0
    for (t_word, t_tag) in word_tag_list_test:
        total_test += 1
        if t_word in predicted_dict:
            if(t_tag in index):
                p_tag = predicted_dict[t_word]
                confusion_matrix[index[p_tag]][index[t_tag]] += 1
                norm = max(norm, confusion_matrix[index[p_tag]][index[t_tag]])
                if p_tag == t_tag:
                    correct += 1
            # else:
                # print("new tag found : " + t_tag)
        else:
            new_word_fount += 1
            p_tag = "NN1"
            confusion_matrix[index[p_tag]][index[t_tag]] += 1
            norm = max(norm, confusion_matrix[index[p_tag]][index[t_tag]])
            if p_tag == t_tag:
                correct += 1

            # print("new word found : " + t_word)
    accuracy = 1.0 * correct / total_test


def create_index_matrix():
    random.shuffle(tagCount_list_train)
    i = 0
    for t in tagCount_list_train:
        index[t[0]] = i
        i += 1


def plot_confusion_matrix():
    row = []
    col = []
    for i in index:
        row.append(i)
        col.append(i)

    # we have confusion matrix as 2d array and row as the list of tags

    normalised = []
    viterbi_normalised = []

    for i in confusion_matrix:
        temp = []
        for j in i:
            temp.append(1.0*j/norm)
        normalised.append(temp)

    for i in viterbi_confusion_matrix:
        temp = []
        for j in i :
            temp.append(1.0 * j / viterbi_norm)
        viterbi_normalised.append(temp)

    df_cm = pd.DataFrame(normalised, index=[i for i in row],
                         columns=[j for j in col])

    plt.figure(figsize=(15, 10.5))
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm)
    plt.title('Confusion matrix')
    plt.savefig("plot.png")

    df_cm = pd.DataFrame(viterbi_normalised, index=[i for i in row],
                         columns=[j for j in col])

    plt.figure(figsize=(15, 10.5))
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm)
    plt.title('Viterbi Confusion matrix')
    plt.savefig("plot_viterbi.png")


def generateB():
    print("generating B for HMM")
    # p(w|t) = count(t|w)/count(t)
    # frequency_list_train contain w,t,count
    # tag_cnt_dict_train contain count of tag

    for x in frequency_list_train:
        w = x[0]
        t = x[1]
        c = x[2]
        tc = tag_cnt_dict_train[t]
        prob = (1.0 * c)/(1.0 * tc)
        B[w, t] = prob


def getB(word, tag):
    if (word, tag) in B:
        return B[word, tag]
    return 0.0


def getA(prev, curr):
    if(prev, curr) in A:
        return (1.0 * A[prev, curr])/(1.0*tag_cnt_dict_train[prev])
    return 0.0


def getPi(tag):
    if tag in Pi:
        return (1.0*Pi[tag])/(1.0 * tot_Pi)
    return 0.0


def viterbi_train(sent):
    global tot_Pi
    n = len(sent)
    for i in range(0, n):
        t = sent[i][1]
        if i == 0:
            tot_Pi += 1
            if t in Pi:
                Pi[t] += 1
            else:
                Pi[t] = 1
        else:
            prev_t = sent[i-1][1]
            if (prev_t, t) in A:
                A[prev_t, t] += 1
            else:
                A[prev_t, t] = 1


def viterbi_test(sent):
    # first we need tags associated with each word
    # for that we can use word_tag_cnt_dict_train
    n = len(sent)
    word = []
    test_tag = []
    for i in sent:
        word.append(i[0])
        test_tag.append(i[1])

    # I think we need to run dp for individual sentence
    dp = {}
    track = {}
    prev = []

    for i in range(0, n):
        tags = get_tags(word[i])
        # min size is 1
        for tag in tags:
            dp[i, tag] = 0.0
        if i == 0:
            for tag in tags:
                p = getPi(tag)
                dp[i, tag] = p
        else:
            for tag in tags:
                for prev_tag in prev:
                    p = dp[i-1, prev_tag] * \
                        getA(prev_tag, tag) * getB(word[i], tag)
                    if(p >= dp[i, tag]):
                        dp[i, tag] = p
                        track[i, tag] = prev_tag

        prev = tags

    calc = []
    max_prob = 0.0
    max_tag = ""
    prev_tag = ""

    if n == 1:
        for tag in get_tags(word[0]):
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

    # calc is the calculated tag sequence
    # test_tag is the actual tag sequence
    global viterbi_correct
    global viterbi_total_test
    global viterbi_norm

    viterbi_total_test += n
    for i in range(0,n):
        viterbi_confusion_matrix[index[calc[i]]][index[test_tag[i]]] += 1
        viterbi_norm = max(viterbi_norm, viterbi_confusion_matrix[index[calc[i]]][index[test_tag[i]]])
        if calc[i]==test_tag[i]:
            viterbi_correct += 1


def get_tags(word):
    val_to_ret = []
    if word in word_tag_cnt_dict_train:
        for tag in word_tag_cnt_dict_train[word]:
            val_to_ret.append(tag)

    if len(val_to_ret) == 0:
        val_to_ret.append("NN1")
    return val_to_ret


def initialise_viterbi_confusion_matrix():
    n = len(tagCount_list_train)
    for _ in range(n):
        temp = []
        for __ in range(n):
            temp.append(0)
        viterbi_confusion_matrix.append(temp)


# MAIN METHOD
print("reading train files....")
word_tag_list_train = readFile(trainFolder)
reading_train = False

create_readable_format()

print("arranging data....")
[word_tag_cnt_dict_train, word_cnt_dict_train,
    tag_cnt_dict_train] = create_dict(word_tag_list_train)

create_frequency_list()

create_word_count_list()

create_tag_count_list()

create_predicted_dictionary()

generateB()

create_index_matrix()

initialise_viterbi_confusion_matrix()

print("reading test files....")
word_tag_list_test = readFile(testFolder)

create_confusion_matrix()

print("accuracy is : ", accuracy)


viterbi_accuracy = (1.0 * viterbi_correct)/(1.0 * viterbi_total_test)
print("viterbi accuracy is : ", viterbi_accuracy)

plot_confusion_matrix()


print("--- %s seconds ---" % (time.time() - start_time))
