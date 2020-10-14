import xml.etree.ElementTree as ET
import csv
import os
import io
from typing import TextIO
from collections import defaultdict


def findFile(path):
    val_to_ret=[]
    for fname in os.listdir(path):
        if os.path.isdir(path+'/'+fname):
            val_to_ret.extend(findFile(path+'/'+fname))
        else:
            val_to_ret.extend(to_parse(path+'/'+fname))
    return val_to_ret

def to_parse(path):
    tree=ET.parse(path)
    rootTree=tree.getroot()
    val_to_ret=[]
    for words in rootTree.iter('w'):
        val_to_ret.append([words.text,words.attrib.get('c5'),words.attrib.get('hw'),words.attrib.get('pos')])

    return val_to_ret

def create_dict(wordList):
    dict_word=defaultdict(dict)
    dict_cnt_c5=defaultdict(int)
    dict_cnt_word=defaultdict(int)
    for x in wordList:
        if x[0] in dict_word:
            if x[1] in dict_word[x[0]]:
                dict_word[x[0]][x[1]]+=1
            else:
                dict_word[x[0]][x[1]]=1
        else:
            dict_word[x[0]][x[1]]=1
        dict_cnt_c5[x[1]]+=1
        dict_cnt_word[x[0]]+=1
    return [dict_word,dict_cnt_word,dict_cnt_c5];


def cmp(aa):
    return aa[1]



def main():

    wordList=findFile('./Train-corups')
    dict_list=create_dict(wordList)
    frequency_table = []
    wordCount=[];
    posCount=[];

    for key in dict_list[0]:
        for c5 in dict_list[0][key]:
            frequency_table.append([key,c5,dict_list[0][key][c5]])
            ##print(key+" "+str(dict_list[0][key][c5]))

    for key in dict_list[1]:
        wordCount.append([key,dict_list[1][key]])

    for key in dict_list[2]:
        posCount.append([key,dict_list[2][key]])

    wordCount.sort(reverse=True,key=cmp)
    posCount.sort(reverse=True, key=cmp)

    file: TextIO
    with io.open('./readableFormat.csv','w',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerows(wordList)

    file: TextIO
    with io.open('./frequencyTable.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(frequency_table)

    file: TextIO
    with io.open('./wordCount.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        for x in range(0,10):
            writer.writerow(wordCount[x])

    file: TextIO
    with io.open('./posCount.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        for x in range(0,10):
            writer.writerow(posCount[x])




if __name__=='__main__':
    main()