# Bag of Words プログラミング

## 1. BoWサンプルプログラムを理解する
1. このプログラムはどこから実行が開始されるか
2. def create_dict(tokens):  はBoW辞書を作成する関数だが、これでどのようにBoW辞書が作成しているのか  
   for recipe in recipes:は何をやっているか。
3. def word_vec( でBoWのデータ部分（単語ベクトル）を作成している。このコーディングの意味を解釈せよ。
4. BoW辞書とBoWデータ部分とを連結してBoWデータセットを完成している部分はどこか  
   
[bow_recipe.py](bow_recipe.py)

``` python 
import pandas as pd
import codecs
import numpy as np

recipes = [['チョコレート', 'バター', '卵', '砂糖', '小麦粉', '好きジャム'],
['生クリーム', 'チョコ', 'バター', '砂糖', '卵', '小麦粉'],
['卵', 'グラニュー', 'チョコレート', '食塩', '小麦粉', '卵白'],
['フォンダンショコラ', 'アイスクリーム', 'イチゴ', 'コイン'],
['チョコレート', '卵', '砂糖', '小麦粉', 'バター', '生地'],
['チョコレート', 'バター', '卵', 'グラニュー', '小麦粉'],
['チョコ', 'バター', 'ココア'],
['チョコレート', '食塩', '卵', '砂糖', '小麦粉'],
['チョコレート', '生クリーム', '酒', 'チョコレート', '生クリーム', '卵黄', '卵白', '食塩', 'グラニュー', 'ココア', 'インスタント']]

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

bow =[]
for recipe in recipes:
    word_vector = word_vec(vocabulary_dic,recipe)    
    bow.append(word_vector)

col = [v for v,i in vocabulary_dic.items()]
bow_df = pd.DataFrame(bow,columns=col)
with codecs.open("./data/recipe_bow.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
```

## 2. 応用演習：csvから文書を読み込んでBoWにする
bow_recipe.pyでは、形態素解析済のデータからBoWを作成したが、以下ではtsukurepo_simple.csvからツクレポのクチコミを1件づつ取り出して、形態素解析をやってからBoWを作成する。  
[bow_tsukurepo.py](bow_tsukurepo.py)の
\###  ###で囲まれた部分に適切なコーディングを書き込んでプログラムを完成させよ。なおコーディングすべき処理は該当箇所に記述している。bow_recipe.pyと比較すると参考になる。  
from tokenizer import tokenizeは、形態素解析の関数。このプログラムを開くと処理内容がわかる。

bow_tsukurepo.pyの中身

``` python
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

tsukurepo_df = pd.read_csv('tsukurepo_simple.csv', encoding='ms932', sep=',',skiprows=0)

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
with codecs.open("tsukurepo_bow.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
```

## 3.CountvectorizerによるBoWの作成
1. 上記2.と同じデータtsukurepo_simple.csvからBoWを作成する。当然、結果は同じになる。  
ただし、Countvectorizerでは、英語は自動的に小文字変換する。そのため、大文字と小文字が混在している場合、小文字に統一される。これが原因で、BoW辞書の次元数は、Countvectorizerの方が１つだけ小さい。

2. 以下のパラメータを追加して低頻度語彙・汎用語彙のフィルタリングをしてみよ。次元数が相当削減されるはず。 
   min_df=0.05, max_df=0.3

3.  [BoW_barChats.py](bow_barChart.py)を参考にして、シュークリーム、プリンそれぞれの棒グラフを表示できるように修正せよ（2.のフィルタリングを行うこと）。注意点は以下。    
   ・import japanize_matplotlib:日本語表示の文字化けを解消する  
   ・シュークリーム、プリンに分けて表示するには予めbow_dfを、それぞれに分けておく必要がある。  
     ・bow_dfから、シュークリーム、プリン　それぞれのベクトルを[条件抽出](https://deepage.net/features/pandas-cond-extraction.html)するには、tsukurepoとbow_dfを[concat](https://deepage.net/features/pandas-concat.html)する。  
 ・上記で条件抽出したDataFrameの平均を計算するには[こちら](https://deepage.net/features/pandas-mean.html#%E5%88%97%E3%81%94%E3%81%A8%E3%81%AE%E5%B9%B3%E5%9D%87%E3%82%92%E6%B1%82%E3%82%81%E3%82%8B)。  
 ・複数のグラフを分割表示するには[subplot](https://stats.biopapyrus.jp/python/subplot.html)



#### 注意点
- bow = vectorizer.transform(texts)で生成されるデータは０要素を省略したもの
- toarray()メソッドは、それを変換して次元数を揃えるもの

``` python

tsukurepo = pd.read_csv('tsukurepo_simple.csv', encoding='ms932', sep=',',skiprows=0)
texts = tsukurepo['tsukurepo']

vectorizer = CountVectorizer(tokenizer=tokenize)  # <2>　引数に形態素解析エンジンを渡す
vec=vectorizer.fit(texts)  
bow = vectorizer.transform(texts)  

print(vec.get_feature_names()) # 見出し（辞書）が表示される
print(bow)# BoWが転置されて表示される。また非ゼロのものだけが表示される
print(bow.toarray()) # これでBoWが表示される


bow_df = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())
```

