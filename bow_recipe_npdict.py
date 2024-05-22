import pandas as pd
import codecs
import numpy as np

recipes = [['チョコレート','バター', '卵', '砂糖', '小麦粉', '好きジャム'],
['生クリーム', 'チョコ', 'バター', '砂糖', '卵', '小麦粉'],
['卵', 'グラニュー', 'チョコレート', '食塩', '小麦粉', '卵白'],
['フォンダンショコラ', 'アイスクリーム', 'イチゴ', 'コイン'],
['チョコレート', '卵', '砂糖', '小麦粉', 'バター', '生地'],
['チョコレート', 'バター', '卵', 'グラニュー', '小麦粉'],
['チョコ', 'バター', 'ココア'],
['チョコレート', '食塩', '卵', '砂糖', '小麦粉'],
['チョコレート', '生クリーム', '酒', 'チョコレート', '生クリーム', '卵黄', '卵白', '食塩', 'グラニュー', 'ココア', 'インスタント']]


def create_dict(tokens):  # <2>
    vocabulary = np.array(list(set(tokens)))   
    
    return vocabulary

def word_vec(vocabulary,review):
    
    # Build BoW Feature Vector <4>
    #word_vector = [0]*len(vocabulary) 
    word_vector = np.zeros(len(vocabulary))    
    for i, word in enumerate(review):       
        
        index = vocabulary[word]
        word_vector[index] += 1

    return word_vector

tokens =[]
for recipe in recipes:
    tokens += recipe  

vocabulary_dic = create_dict(tokens)
print(vocabulary_dic)
bow =[]
for recipe in recipes:
    word_vector = word_vec(vocabulary_dic,recipe)    
    bow.append(word_vector)

col = [v for v,i in vocabulary_dic.items()]
bow_df = pd.DataFrame(bow,columns=col)
with codecs.open("./data/recipe_bow.csv", "w", "ms932", "ignore") as f:   
    bow_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
