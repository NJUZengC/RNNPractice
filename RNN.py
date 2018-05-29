# -*- coding: utf-8 -*-
# @Time    : 2018/5/28 20:48
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : RNN.py
# @Software: PyCharm Community Edition
import numpy as np
import matplotlib.pyplot as plt
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

class RNN(object):

    def __init__(self,input_dim,hidden_dim=100,bptt_truncate=4,learning_rate = 0.05):


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.learning_rate = learning_rate
        self.U = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (input_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward(self,x):

        T = len(x)
        s = np.zeros((T+1, self.hidden_dim))
        o = np.zeros((T,self.input_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U.dot(x[t]) + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        return [o,s]

    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward(x[i])
            o = np.log(o)

            L += -1 * np.sum(o * y[i])

        return L

    def backword(self,x,y):

        T = len(x)
        o,s = self.forward(x)
        #print(y)
        #print(o)

        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdV = np.zeros(self.V.shape)

        delta_o = o
        delta_o -= y
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t],s[t])

            delta_t = self.V.T.dot(delta_o[t]) * (1 - s[t] ** 2)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU += np.outer(delta_t, x[bptt_step])

                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)

        return [dLdU,dLdV,dLdW]

    def update(self,dLdU,dLdV,dLdW):
        self.U -=  self.learning_rate * dLdU
        self.V -= self.learning_rate * dLdV
        self.W -= self.learning_rate * dLdW

    def train(self,x,y,max_Term = 100):
        lens = len(x)
        L = []
        Terms = []
        for i in np.arange(max_Term):
            if i%10 == 0:
                print("term finished %i" %i)
            for j in np.arange(lens):
                dLdU, dLdV, dLdW = self.backword(x[j],y[j])
                self.update(dLdU,dLdV,dLdW)
            L.append(self.calculate_total_loss(x,y))
            Terms.append(i)
        return [L,Terms]

    def predict(self,x):
        o,_ = self.forward(x)
        index = []
        for i in range(len(o)):
            maxindex = 0
            maxv = 0
            for j in range(len(o[i])):
                if o[i][j] > maxv:
                    maxv = o[i][j]
                    maxindex = j
            index.append(maxindex)
        return index

    def plot(self,L,Terms):
        plt.plot(Terms,L)
        plt.show()

def generate(word_dim,list_dim,sample_num):
    X = []
    Y = []
    np.random.seed(0)
    for i in range(sample_num):
        x = list(np.random.randint(word_dim, size=list_dim) - 1)
        y = x[1:]
        y.append(0)
        for j in range(list_dim):
            temp = np.zeros(word_dim)
            temp[x[j]] = 1
            x[j] = temp
            temp = np.zeros(word_dim)
            temp[y[j]] = 1
            y[j] = temp
        X.append(x)
        Y.append(y)
        #print(len(X[i]))
        #print(len(Y[i]))
    return [X,Y]



if __name__ == '__main__':

    X,Y = generate(10,5,100)
    #X = [np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]]),np.array([[0,0,1,0,0,0],[0,1,0,0,0,0]])]
    #Y = [np.array([[0,1,0,0,0,0],[0,0,1,0,0,0]]),np.array([[0,1,0,0,0,0],[1,0,0,0,0,0]])]

    rnn = RNN(10,5,4,learning_rate=0.005)
    L,Terms = rnn.train(X,Y,300)
    rnn.plot(L,Terms)
    #rnn.predict(np.array([[0,0,0,1,0,0]]))

