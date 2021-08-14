import numpy as np

# 数据处理
# 输出也归一化，并且输出也假设sigmod函数
def loaddataset(filename):
    fp = np.loadtxt(filename, delimiter=",",  usecols=(0, 1, 2, 3))
    dataset_1 = fp[:, 0:3]
    labelset_1 = fp[:, 3:4]
    # 数据归一化
    maxcols_1 = dataset_1.max(axis=0)
    mincols_1 = dataset_1.min(axis=0)
    data_shape_1 = dataset_1.shape
    data_rows_1 = data_shape_1[0]
    data_cols_1 = data_shape_1[1]
    dataset = np.empty((data_rows_1, data_cols_1))
    for i in range(data_cols_1-1):
        dataset[:, i] = (dataset_1[:, i] - mincols_1[i]) / (maxcols_1[i] - mincols_1[i])
    maxcols_2 = labelset_1.max(axis=0)
    mincols_2 = labelset_1.min(axis=0)
    data_shape_2 = labelset_1.shape
    data_rows_2 = data_shape_2[0]
    data_cols_2 = data_shape_2[1]
    labelset = np.empty((data_rows_2, data_cols_2))
    for i in range(data_cols_2):
        labelset[:, i] = (labelset_1[:, i] - mincols_2[i]) / (maxcols_2[i] - mincols_2[i])
    return dataset, labelset
# 权重与阈值建立
# x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
# 创建的是参数初始化函数，参数有各层间的权重weight和阈值即偏置value就是b


def parameter_initialization(x, y, z):
    weight1 = np.random.randint(0, 1, (x, y)).astype(np.float64)
    weight2 = np.random.randint(0, 1, (y, z)).astype(np.float64)
    value1 = np.random.randint(0, 1, (1, y)).astype(np.float64)
    value2 = np.random.randint(0, 1, (1, z)).astype(np.float64)
    return weight1, weight2, value1, value2

# 创建激活函数sigmoid


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 创建训练样本的函数，返回训练完成后的参数weight和value，这里的函数是经过一次迭代后的参数，即所有的样本经过一次训练后的参数
# 具体参数的值可以通过设置迭代次数和允许误差来进行确定
# 前向计算
def trainning(dataset, labelset, weight1, weight2, value1, value2):
    # x为步长
    x = 0.0001
    # 学习率
    for i in range(len(dataset)):    # 依次读取数据特征集中的元素，一个元素即为一个样本所含有的所有特征数据
        inputset = np.mat(dataset[i]).astype(np.float64)
        # print(inputset)
        outputset = np.mat(labelset[i]).astype(np.float64)
        # print(weight1)
        input1 = np.dot(inputset, weight1).astype(np.float64)
        # print(input1)
        h = sigmoid(input1 + value1).astype(np.float64)
        # print(h)
        input2 = np.dot(h, weight2).astype(np.float64)
        o = input2 + value2
        g = (o - outputset)  # loss
        b = np.dot(g, np.transpose(weight2))
        # print(b)
        c = np.multiply(h, 1 - h)
        # print(c)
        e = np.multiply(b, c)
        value1_change = -x * e
        # print(value1_change.shape)
        value2_change = -x * g
        weight1_change = -x * np.dot(np.transpose(inputset), e)
        # print(weight1_change)
        weight2_change = -x * np.dot(np.transpose(h), g)
        # 更新参数，权重与阈值的迭代公式
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change
    return weight1, weight2, value1, value2
# 创建测试样本数据的函数


def testing(dataset1, labelset1, weight1, weight2, value1, value2):
    totalerror = 0
    output = open('0.5_huatu_0.9.dat', 'w', encoding='gbk')
    fp = np.loadtxt('TEST_0.9.csv', delimiter=",", usecols=(0, 1, 2, 3))
    for i in range(len(dataset1)):
        # 计算每一个样例的标签通过上面创建的神经网络模型后的预测值
        inputset = dataset1[i]
        outputset = labelset1[i]
        # 反归一化
        h = sigmoid(np.dot(inputset, weight1) + value1)
        o = np.dot(h, weight2) + value2
        output.write(str(inputset[0]) + str(',') + str(o[0][0]))
        output.write('\n')
        errorrate = (outputset - o) / outputset
        # print(errorrate)
        totalerror = totalerror + abs(errorrate)
    error = totalerror / len(dataset1)
    print(error)

def main():
    # 读取训练样本数据并且进行样本划分
    dataset, labelset = loaddataset('train_total_5.csv')
    # 读取测试样本数据并且进行样本划分
    dataset1, labelset1 = loaddataset('TEST_0.9.csv')
    # 得到初始化的待估参数的值
    weight1, weight2, value1, value2 = parameter_initialization(len(dataset[0]), 12, 1)
    # 迭代次数为1500次，迭代次数一般越大准确率越高，但是其运行时间也会增加
    for i in range(400):
        # 获得对所有训练样本训练迭代一次后的待估参数
        weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
    # 对测试样本进行测试，并且得到正确率
    error = testing(dataset1, labelset1, weight1, weight2, value1, value2)
    print(error)
    print(weight1, weight2, value1, value2)


if __name__ == '__main__':
    main()