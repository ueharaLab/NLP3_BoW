import MeCab

tagger = MeCab.Tagger()


def tokenize(texts):
    token = tagger.parseToNode(texts)

    word_list = []
    while token:
            #print(token.surface,token.feature)   
        hinshi = token.feature.split(',')
        if hinshi[0] =='名詞' and hinshi[1] =='一般' :
            word_list.append(token.surface)
        token = token.next
    return word_list

    
