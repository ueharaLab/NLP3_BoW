import pandas as pd
import numpy as np
import MeCab
import collections
import japanize_matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 30,10

tagger = MeCab.Tagger("mecabrc -u c:/neologd/NEologd.dic")
tagger.parse('')# 実行の準備

tabelog_df = pd.read_csv('./data/tabelog.csv', encoding='ms932', sep=',',skiprows=0)

def mecab_tokenizer(reviews):

    word_list=[]    
    token=tagger.parseToNode(reviews)
    
    while token:
        #print(token.surface,token.feature)   
        hinshi = token.feature.split(',')
        if hinshi[0] =='名詞' and hinshi[1] =='一般' and token.surface != 'うどん' and token.surface !='ラーメン' and token.surface.find('店')== -1:
            word_list.append(token.surface)
        token = token.next
    return word_list


def bar_data(word_list):
	
    word_counter = collections.Counter(word_list)
    count_list=[]
    word_list=[]
    for k,v in word_counter.items():

        if v >15 :
            word_list.append(k)
            count_list.append(v)
    return word_list,count_list


udon =[]
ramen =[]
for i,row in tabelog_df.iterrows():

    words = mecab_tokenizer(row['text'])
    if row['ジャンル'] == 'うどん':
        udon+=words
    elif row['ジャンル'] == 'ラーメン':
        ramen+=words
		
udon_key,udon_val = bar_data(udon)
ramen_key, ramen_val = bar_data(ramen)

fig = plt.figure()
ax1=fig.add_subplot(211,title='うどん棒グラフ')
ax1.bar(np.arange(len(udon_val)), np.array(udon_val), tick_label=udon_key, align="center")
ax1.set_xticks(np.arange(len(udon_val)))
ax1.set_xticklabels(udon_key, rotation=45, ha='right',fontsize=14)
#ax1.xticks(np.arange(len(udon_val)),udon_key)
ax2=fig.add_subplot(212,title='ラーメン棒グラフ')
ax2.bar(np.arange(len(ramen_val)), np.array(ramen_val), tick_label=ramen_key, align="center")
ax2.set_xticks(np.arange(len(ramen_val)))
ax2.set_xticklabels(ramen_key, rotation=45, ha='right',fontsize=14)
#ax1.xticks(np.arange(len(ramen_val)),ramen_key)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.show()
