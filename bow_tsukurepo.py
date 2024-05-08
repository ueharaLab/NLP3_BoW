import pandas as pd
import codecs
import numpy as np
from tokenizer import tokenize


tsukurepo_df = pd.read_csv('./data/tsukurepo_simple.csv', encoding='ms932', sep=',',skiprows=0)

recipes =[]
###
1. tsukurepo_dfから、'tsukurepo'の文書を1行づつ読み込む
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

2. 1行毎に形態素解析して語彙のリストを作成する（形態素解析は外部の関数 tokenizerを使う。
   この関数は形態素解析した結果（単語列）をリスト型で返却する）
3. 2.の結果をrecipesに加える
###


def create_dict(tokens):  # <2>
    # Build vocabulary <3>
    vocabulary = {}

    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
            #print(vocabulary)
    return vocabulary

def word_vec(vocabulary,review):
    
    # Build BoW Feature Vector <4>
    word_vector = [0]*len(vocabulary)     
    for i, word in enumerate(review):
       
        index = vocabulary[word]
        word_vector[index] += 1

    return word_vector


tokens =[]
for recipe in recipes:
    tokens += recipe 
 

vocabulary_dic = create_dict(tokens)
#print(vocabulary_dic)

bow =[]
for recipe in recipes:
    word_vector = word_vec(vocabulary_dic,recipe)    
    bow.append(word_vector)


col = [v for v,i in vocabulary_dic.items()]
bow_df = pd.DataFrame(bow,columns=col)
with codecs.open("./data/tsukurepo_bow.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
