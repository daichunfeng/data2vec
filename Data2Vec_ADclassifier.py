import numpy as np
import os
import librosa
from transformers import AutoProcessor, AutoModelForCTC
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as Fun
import torch.nn as nn
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import re

processor = AutoProcessor.from_pretrained("/root/.cache/huggingface/transformers/data2vec-audio-base-960h")
model = AutoModelForCTC.from_pretrained("/root/.cache/huggingface/transformers/data2vec-audio-base-960h")
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = torch.nn.Linear(14976, 5000)
#         self.fc2 = torch.nn.Linear(5000,1000)
#         self.fc3 = torch.nn.Linear(1000,500)
#         self.fc4 = torch.nn.Linear(500, 20)
#         self.fc5 = torch.nn.Linear(20, 2)
#
#     def forward(self, x):
#         x = Fun.relu(self.fc1(x))
#         x = Fun.relu(self.fc2(x))
#         x = Fun.relu(self.fc3(x))
#         x = Fun.relu(self.fc4(x))
#         m = nn.Softmax(dim=1)
#         x = m(self.fc5(x))   # x = nn.Softmax(self.fc4(x),dim=1)     # m = nn.Softmax(dim=1)# outputs_softmax = m(outputs)
#         return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(700, 350)
        self.fc2 = torch.nn.Linear(350,100)
        self.fc3 = torch.nn.Linear(100,20)
        self.fc4 = torch.nn.Linear(20,2)

    def forward(self, x):
        x = Fun.relu(self.fc1(x))
        x = Fun.relu(self.fc2(x))
        x = Fun.relu(self.fc3(x))
        # x = Fun.relu(self.fc4(x))
        m = nn.Softmax(dim=1)
        x = m(self.fc4(x))   # x = nn.Softmax(self.fc4(x),dim=1)     # m = nn.Softmax(dim=1)# outputs_softmax = m(outputs)
        return x
def train(batch_size,num_epochs,lr):
    net=Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    t_list = []
    loss_list = []
    t = 0
    num_epochs = num_epochs
    i_list, l_list = data_iter(batch_size, Xtrain, Ytrain)
    for epochs in range(1, num_epochs + 1):
        for X, Y in zip(i_list, l_list):
            out = net(X)  # 输入input,输出out
            loss = loss_func(out, Y.squeeze())  # 输出与label对比
            optimizer.zero_grad()
            loss.backward()  # 前馈操作
            optimizer.step() # 使用梯度优化器
            loss_list.append(float(loss.detach().numpy()))
            t = t + 1
            t_list.append(t)
            # if t%5==0:
            #     print('iteration',t)
        # torch.save(net,"trained_model.pt")
        # print(loss_list)
    return net,t_list,loss_list
def predict(net, Xtest, Ytest):
    list1 = []
    t = 0
    # Xtest, Ytest = Smote(Xtest, Ytest)
    with torch.no_grad():
        out = net(Xtest)  # 输入input,输出out
    a = torch.argmax(out, dim=-1).numpy()
    y = a.ravel()
    print(y)
    Ytest = Ytest.numpy()
    acuracy = float((y == Ytest).astype(int).sum()) / float(Ytest.size)
    print("ADcontest acuracy",acuracy)
    # for i in range(Ytest.shape[0]):
    #     with torch.no_grad():
    #         out = net(Xtest[i].unsqueeze(-1).T)
    #     list1.append(torch.argmax(out['logits']).item())
    #     if i%50 ==0:
    #         print(i)
    # y = np.array(list1)
    # print(y)
    # acuracy = float((y == Ytest).astype(int).sum()) / float(Ytest.size)
    # print("ADcontest acuracy", acuracy)
    return acuracy
def data_iter(batch_size, inputs, labels):
    i_list =[]
    l_list =[]
    num_examples = inputs.shape[0]
    for i in range(0, num_examples, batch_size):
        # namesx = str(i) +'th_part_of_inputs'
        # namesy = str(i) + 'th_part_of_labels'
        # pathx = os.path.join('/home/pythonProject/Xiterated',namesx)
        # pathy = os.path.join('/home/pythonProject/Xiterated', namesy)
        # torch.save(inputs[i: min(i + batch_size, num_examples)],pathx)
        # torch.save(labels[i: min(i + batch_size, num_examples)],pathy)
        # i_list.append(torch.load('')
        i_list.append(inputs[i: min(i + batch_size, num_examples)])
        l_list.append(labels[i: min(i + batch_size, num_examples)])
    return i_list,l_list
def plot_line(list1,list2):
     plt.plot(list1,list2)
     plt.show()
def random_shuffle(data,label):
  dataset=np.hstack([data,label])
  np.random.seed(12345)
  np.random.shuffle(dataset)
  return dataset[:,:-1],dataset[:,-1]
def Smote(x,y):
    smote = SMOTE(random_state=1)
    Xtrain, Ytrain = smote.fit_resample(x, y)
    Xtrain = torch.tensor(np.float64(Xtrain), dtype=torch.float32)
    Ytrain = torch.LongTensor(Ytrain)
    return Xtrain,Ytrain

def pca():
    pca = PCA(700)  # 实例化
    X = torch.load('FInalXfor813.pt')
    pca = pca.fit_transform(X.detach().numpy())  # 拟合模型
    X = torch.tensor(np.float64(pca), dtype=torch.float32)
    return X
def KfoldX_Y():
    # X = torch.load('FInalXfor813.pt')
    X = pca()
    Y = torch.LongTensor(np.load('label_list813.npy'))
    Y = Y.numpy().tolist()
    X, Y = Smote(X, Y)
    a = np.arange(0,700)
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    listXtrain = []
    listYtrain = []
    listXtest = []
    listYtest = []
    for i in range(10):
        label1 = []
        label2 = []
        index_train,index_test = [*kf.split(a)][i][0],[*kf.split(a)][i][1]
        np.random.shuffle(index_test)
        np.random.shuffle(index_train)
        tensor1 = torch.empty(1,700)
        tensor2 = torch.empty(1,700)
        for b in index_train:
            tensor1  = torch.cat([tensor1,X[b].view(1,-1)],dim = 0)
            label1.append(Y[b])
        listXtrain.append(tensor1[1:])
        listYtrain.append(torch.LongTensor(np.array(label1)))
        for c in index_test:
            tensor2  = torch.cat([tensor2,X[c].view(1,-1)],dim = 0)
            label2.append(Y[c])
        listXtest.append(tensor2[1:])
        listYtest.append(torch.LongTensor(np.array(label2)))

    return listXtrain,listXtest,listYtrain,listYtest


if __name__=='__main__':
    # x = audio2vec('/home/pythonProject/audiofolder/data')
    listXtrain, listXtest, listYtrain, listYtest = KfoldX_Y()
    t = 0
    accuracy_list = []
    for Xtrain, Xtest, Ytrain, Ytest in zip(listXtrain, listXtest, listYtrain, listYtest):
        t += 1
        print('this is the %d th fold' % (t)) # 解决样本不均衡
        net, t_list, loss_list = train(batch_size=30, num_epochs=2, lr=0.001)
        acuracy = predict(net, Xtest, Ytest)
        accuracy_list.append(acuracy)
    print(np.array(accuracy_list).mean())