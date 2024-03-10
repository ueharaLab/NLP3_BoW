import pandas as pd
import codecs
import numpy as np
from tokenizer import tokenize

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

tsukurepo_df = pd.read_csv('./data/tsukurepo_simple.csv', encoding='ms932', sep=',',skiprows=0)

tokens =[]
###
tsukurepo_dfから'tsukurepo'の文書（口コミ）を1件づつ
取り出して形態素解析する。
（形態素解析は、外部の関数 tokenizerを使う。この関数は
　形態素解析した結果（単語列）をリスト型で返却する）
形態素解析した結果をtokensの要素にする
###
vocabulary_dic = create_dict(tokens)

bow =[]
###
以下の処理を文書（口コミ）毎に繰り返す

1) tsukurepo_dfから'tsukurepo'の文書（口コミ）を1件づつ
取り出して形態素解析する。
（形態素解析は、外部の関数 tokenizerを使う。この関数は
　形態素解析した結果（単語列）をリスト型で返却する）

2) 形態素のリストを単語ベクトルに変換する

3) bowに単語ベクトルを追加する
###

col = [v for v,i in vocabulary_dic.items()]
bow_df = pd.DataFrame(bow,columns=col)
with codecs.open("./data/tsukurepo_bow.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
