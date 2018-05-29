# -*- coding: utf-8 -*-
# @Time    : 2018/5/29 9:28
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : generateText.py
# @Software: PyCharm Community Edition
import numpy as np
from RNN import RNN
import random
def read(path):
    x = []
    y = []
    max_line = 0
    dict = {}
    with open(path,encoding='utf-8') as f:
        index = 2
        for line in f:
            if len(line)<2:
                continue
            line_x = [1]
            for word in line:
                if word in dict.keys():
                    line_x.append(dict[word])
                else:
                    dict[word] = index
                    index += 1
                    line_x.append(dict[word])
            line_x.append(0)
            if max_line < len(line_x):
                max_line = len(line_x)
            x.append(line_x)
            y.append(line_x[1:])

    for i in range(len(x)):
        if len(x[i])<max_line:
            x[i].extend([0] * (max_line - len(x[i])))
        if len(y[i]) < max_line:
            y[i].extend([0] * (max_line - len(y[i])))
    print(x[0])
    for i in range(len(x)):

        for j in range(len(x[i])):
            temp  = [0] * index
            temp[x[i][j]] = 1
            x[i][j] = temp
        for j in range(len(y[i])):
            temp  = [0] * index
            temp[y[i][j]] = 1
            y[i][j] = temp
    return [x,y,index,dict]




if __name__ == '__main__':

    X,Y,index,dicts = read('1168.txt')
    print("read finished")
    print(index)
    rnn = RNN(index, 20)

    rnn.train(X, Y)
    s = [0] * index
    s[1] = 1
    predict_x = []
    for i in range(5):
        temp = [0] * index
        p = random.randint(2,index-1)
        temp[p] = 1
        predict_x.append(temp)
    #index = rnn.predict(np.array([s,predict_x[0],predict_x[1],predict_x[2],predict_x[3],predict_x[4]]))
    index = rnn.predict(np.array([s, predict_x[0], predict_x[1], predict_x[2]]))
    print(index)
    for i in index:
        for j in dicts.keys():

            if dicts[j] == i:
                print(j)