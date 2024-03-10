from tokenizer import tokenize  
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import codecs


tsukurepo = pd.read_csv('./data/tsukurepo_simple.csv', encoding='ms932', sep=',',skiprows=0)
texts = tsukurepo['tsukurepo']

vectorizer = CountVectorizer(tokenizer=tokenize)  # <2>　引数に形態素解析エンジンを渡す
vec=vectorizer.fit(texts)  
bow = vectorizer.transform(texts) 

print(vec.get_feature_names()) # 見出し（辞書）が表示される
print(bow)# BoWが転置されて表示される。また非ゼロのものだけが表示される
print(bow.toarray()) # これでBoWが表示される


bow_df = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())
print(bow_df)

with codecs.open("./data/tsukurepo_bow_vectorizer.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
