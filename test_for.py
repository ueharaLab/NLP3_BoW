import numpy as np
dic = {'a':0,'b':1,'c':2,'d':3}
midashi=[]
for v,i in dic.items():
    midashi.append(v)
print(midashi)

midashi = [v for v,i in dic.items()]
print(midashi)
