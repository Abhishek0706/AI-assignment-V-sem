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

trainFolder = './Train-corups/A1'
testFolder = './Test-corpus/AN'
frequency_list_train = []
wordCount_list_train = []
tagCount_list_train = []
word_tag_list_train = []
confusion_matrix = []
word_tag_list_test = []
word_tag_cnt_dict_train = {}
word_cnt_dict_train = {}
tag_cnt_dict_train = {}
predicted_dict = {}
index = {}

correct = 0
total_test = 0
accuracy = 0


def readFile(path):
    val_to_ret = []
    for fname in os.listdir(path):
        if os.path.isdir(os.path.join(path,fname)):
            val_to_ret.extend(readFile(os.path.join(path,fname)))
        else:
            val_to_ret.extend(parseFile(os.path.join(path,fname)))
    return val_to_ret

def parseFile(path):
    print(path)
    tree = ET.parse(path)
    rootTree = tree.getroot()
    val_to_ret = []
    for words in rootTree.iter('w'):
        li = list(words.attrib.get('c5').split("-"))
        for c5word in li:
            try:
                val_to_ret.append([words.text.strip(), c5word.strip()])
            except:
                print(words.text,c5word)

    for words in rootTree.iter('c'):
        li = list(words.attrib.get('c5').split("-"))
        for c5word in li:
            try:
                val_to_ret.append([words.text.strip(), c5word.strip()])
            except:
                print(words.text,c5word)

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
            frequency_list_train.append([word, tag, word_tag_cnt_dict_train[word][tag]])
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
            if word_tag_cnt_dict_train[word][tag] > maxi : 
                maxi = word_tag_cnt_dict_train[word][tag]
                maxi_tag = tag
        predicted_dict[word] = maxi_tag

def create_confusion_matrix():
    global total_test
    global correct
    print("creating confusion matrix....")
    random.shuffle(tagCount_list_train)
    # initialising confusion matrix
    n = len(tagCount_list_train)
    for _ in range(n):
        temp = []
        for __ in range(n):
            temp.append(0);
        confusion_matrix.append(temp)

    # mapping tag to index 
    i = 0
    for t in tagCount_list_train:
        index[t[0]] = i
        i += 1
    
    new_word_fount = 0
    for (t_word,t_tag) in word_tag_list_test:
        total_test += 1
        if t_word in predicted_dict:
            if(t_tag in index):
                p_tag = predicted_dict[t_word]
                confusion_matrix[index[p_tag]][index[t_tag]] += 1
                if p_tag == t_tag:
                    correct += 1
            # else:
                # print("new tag found : " + t_tag)
        else:
            new_word_fount += 1
            # print("new word found : " + t_word)
    print("new word found are", end=" : ")
    print(new_word_fount)

def plot_confusion_matrix():
    row = []
    col = []
    for i in index:
        row.append(i)
        col.append(i)

    # we have confusion matrix as 2d array and row as the list of tags
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in row],
                    columns = [j for j in col])


    plt.figure(figsize=(15, 10.5))
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm)
    plt.title('Confusion matrix')
    plt.show()

# MAIN METHOD
print("reading train files....")
word_tag_list_train = readFile(trainFolder)

create_readable_format()

print("arranging data....")
[word_tag_cnt_dict_train, word_cnt_dict_train, tag_cnt_dict_train] = create_dict(word_tag_list_train)

create_frequency_list()

create_word_count_list()

create_tag_count_list()

print("reading test files....")
word_tag_list_test = readFile(testFolder)

create_predicted_dictionary()

create_confusion_matrix()

plot_confusion_matrix()

