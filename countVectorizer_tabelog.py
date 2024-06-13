from tokenizer import tokenize  
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import codecs


tabelog = pd.read_csv('./data/tabelog.csv', encoding='ms932', sep=',',skiprows=0)

'''
countVectorizerを使ってtabelogのtextsをBoWに変換するコーディングを書く

'''


with codecs.open("./data/tabelog_bow_vectorizer.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
