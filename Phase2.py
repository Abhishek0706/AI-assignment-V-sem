import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

trainFolder = './Train-corups'
testFolder = './Test-corpus'

word_tag_count_dict = {}
tag_dict = {}
max_dict = {}
confusion_matrix = {}

correct = 0
total_test = 0


def train_util(w_c, tag):
    w_c = w_c.strip()
    tag = tag.strip()

    if(w_c not in word_tag_count_dict):
        word_tag_count_dict[w_c] = {tag: 1}
    else:
        if(tag not in word_tag_count_dict[w_c]):
            word_tag_count_dict[w_c][tag] = 1
        else:
            word_tag_count_dict[w_c][tag] += 1

    if(tag not in tag_dict):
        tag_dict[tag] = 1
    else:
        tag_dict[tag] += 1


def test_util(w_c, tags):
    global correct, total_test, confusion_matrix
    total_test += 1

    try:
        w_c = w_c.strip()
        for tag in tags:
            tag = tag.strip()
    except:
        print(w_c, tags)

    if w_c in max_dict:
        predicted = max_dict[w_c]
        if(predicted in tags):
            correct += 1
            confusion_matrix[predicted][predicted] += 1
        else:
            confusion_matrix[predicted][tags[0]] += 1
    else:
        confusion_matrix['NN1'][tags[0]] += 1


print('Reading train folder...')
for root, dirs, files in os.walk(trainFolder):
    for file in files:
        filepath = os.path.join(root, file)
        tree = ET.parse(filepath)
        treeroot = tree.getroot()
        for w in treeroot.iter('w'):
            word = w.text
            tags = w.attrib['c5'].split('-')
            for tag in tags:
                train_util(word, tag)

        for c in treeroot.iter('c'):
            char = c.text
            tags = c.attrib['c5'].split('-')
            for tag in tags:
                train_util(char, tag)

        for mw in treeroot.iter('mw'):
            mwtext = ''
            for w in mw:
                mwtext += w.text
            tags = mw.attrib['c5'].split('-')
            for tag in tags:
                train_util(mwtext, tag)

print('Creating max dictionary...')
for word, tag_count_dict in word_tag_count_dict.items():
    max_dict[word] = max(tag_count_dict, key=tag_count_dict.get)

print('Initializing confusion matrix...')
for tag in tag_dict:
    confusion_matrix[tag] = {}
    for tagg in tag_dict:
        confusion_matrix[tag][tagg] = 0

print('Reading test folder...')
for root, dirs, files in os.walk(testFolder):
    for file in files:
        filepath = os.path.join(root, file)
        tree = ET.parse(filepath)
        treeroot = tree.getroot()
        for w in treeroot.iter('w'):
            word = w.text
            tags = w.attrib['c5'].split('-')
            test_util(word, tags)

        for c in treeroot.iter('c'):
            char = c.text
            tags = c.attrib['c5'].split('-')
            test_util(char, tags)

        for mw in treeroot.iter('mw'):
            mwtext = ''
            for w in mw:
                mwtext += w.text
            tags = mw.attrib['c5'].split('-')
            test_util(mwtext, tags)

print(correct, 'correct predictions out of', total_test, 'total')
print('Accuracy:', float(correct)/(total_test))

print('Normalizing confusion matrix...')
numpy_array = np.array([[confusion_matrix[predicted][actual] for actual in confusion_matrix[predicted]] for predicted in confusion_matrix])
row_sums = numpy_array.sum(axis=1)
normalized = numpy_array / row_sums[:, np.newaxis]
for i in range(len(normalized)):
    if(np.isnan(normalized[i][0])):
        normalized[i] = np.zeros(61)

print('Plotting confusion matrix...')
df_cm = pd.DataFrame(normalized, confusion_matrix.keys(), confusion_matrix.keys())
plt.figure(figsize=(15, 10.5))
sn.set(font_scale=0.8)
sn.heatmap(df_cm)
plt.title('Confusion matrix')
plt.show()
