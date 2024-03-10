
import pandas as pd
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 30,10

tabelog_bow = pd.read_csv('./data/tabelog_bow.csv', encoding='ms932', sep=',',skiprows=0)
c1 =tabelog_bow[tabelog_bow['ジャンル']=='うどん']
c2 =tabelog_bow[tabelog_bow['ジャンル']=='ラーメン'] 

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

colors =['red','blue']
# ３次元の散布図を描画。戻り値としてmappableオブジェクトを取得。
ax.scatter(c1['コシ'],c1['醤油'],c1['スープ'], c=colors[0],s=100,alpha=0.3,label='うどん')
ax.scatter(c2['コシ'],c2['醤油'],c2['スープ'], c=colors[1],s=100,alpha=0.3,label='ラーメン')
ax.set_xlabel("コシ")
ax.set_ylabel("醤油")
ax.set_zlabel("スープ")
plt.legend()
plt.show()

         