import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def loaddataset(filename):
    fp = np.loadtxt(filename, delimiter=",",  usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    # (np.random.shuffle(fp))
    dataset_1 = fp[:, 0:7]
    labelset = fp[:, 7:8]
    # 数据归一化
    maxcols = dataset_1.max(axis=0)
    mincols = dataset_1.min(axis=0)
    data_shape = dataset_1.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    for i in range(data_cols-5):
        dataset_1[:, i] = (dataset_1[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    # print(dataset_1)
    return dataset_1, labelset


dataset, labelset = loaddataset('0.9_tran.csv')
dataset1, labelset1 = loaddataset('0.9_test.csv')


class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out =self.predict(out)

        return out

net = Net(7,10,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.09)
loss_func = torch.nn.MSELoss()
for t in range(8000):
    prediction = net(torch.from_numpy(np.array(dataset)).float())
    loss = loss_func(prediction,torch.from_numpy(np.array(labelset)).float())
    # print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # output = open('1111_data.dat', 'w', encoding='gbk')
    pre_y = net(torch.from_numpy(np.array(dataset1)).float())
    aa = pre_y.tolist()
    bb = labelset1 - aa
    lama = pre_y.detach().numpy()
    np.savetxt("0.9_fai.txt", lama)
    # output.write(str(lama))
    # output.write('\n')
