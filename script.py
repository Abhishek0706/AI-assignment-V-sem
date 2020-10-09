import xml.etree.ElementTree as ET
import csv
import os
import io
from typing import TextIO


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
    ##print(val_to_ret)
    return val_to_ret

def main():
    file: TextIO
    with io.open('./readableFormat.csv','w',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerows(findFile('./Train-corups'))


if __name__=='__main__':
    main()