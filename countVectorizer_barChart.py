from tokenizer import tokenize  
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import codecs
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

tsukurepo = pd.read_csv('./data/tsukurepo_simple.csv', encoding='ms932', sep=',',skiprows=0)
texts = tsukurepo['tsukurepo']

vectorizer = CountVectorizer(tokenizer=tokenize,min_df=0.05, max_df=0.3)  # <2>　引数に形態素解析エンジンを渡す
vec=vectorizer.fit(texts)  
bow = vectorizer.transform(texts) 

print(vec.get_feature_names()) # 見出し（辞書）が表示される
print(bow)# BoWが転置されて表示される。また非ゼロのものだけが表示される
print(bow.toarray()) # これでBoWが表示される


bow_df = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())


with codecs.open("./data/tsukurepo_bow_vectorizer.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)

bow_df=pd.concat([tsukurepo['keyword'],bow_df],axis=1)
column_name = bow_df.columns[1:]
syu = bow_df[bow_df['keyword']=='シュークリーム'].iloc[:,1:]
syu_mean = [v for k,v in syu.mean().items()]
'''
###
1.プリンのデータのみbow_dfからbowを抽出する
2. このbowの列毎の平均値をベクトル化してリストにする
###
'''
fig = plt.figure()
ax1=fig.add_subplot(211,title='シュークリーム')
ax1.bar(np.arange(len(syu_mean)), np.array(syu_mean), tick_label=column_name, align="center")
ax1.set_xticks(np.arange(len(syu_mean)))
ax1.set_xticklabels(column_name, rotation=45, ha='right',fontsize=14)
#ax1.xticks(np.arange(len(udon_val)),udon_key)
ax2=fig.add_subplot(212,title='プリン')
'''
###
上記シュークリームの棒グラフと同様に、プリンのbow平均を棒グラフにする
###
'''
#ax1.xticks(np.arange(len(ramen_val)),ramen_key)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.show()

