# -*- coding:GBK -*-
import pandas as pd
import numpy
import numpy as np
from sklearn.cluster import k_means
import csv
import sklearn.neighbors.typedefs
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
'''
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#降维
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
'''
#想要聚成几类？考虑到可视化应<8
GroupsYouWant = 5
#考察的指标的数量，即数据中的列数，需要和下面权重的个数一致
ColumnsInvolved = 7
#权重：业务量	服务人次	不满意次数	工作占空比	平均办理时间	业务办理效率	办理超时率
WeightOfColumns = [0.364121667,0.170238008,0.107485731,0.103761228,0.091030417,0.091030417,0.072332533] #广州不转置
WeightOfColumns = [0.042978215,0.080374517,0.145101754,0.155178897,0.17688102,0.17688102,0.222604578] #广州转置
#WeightOfColumns = 1.0/7 * np.array([1,1,1,1,1,1,1]) #不加权

#评KPI时需要反转的列号（从1开始）
Inverse = [5,7]
#评价KPI时需要z_score的列号
Z_score = [1,2,4,6]
#1-0.05 * N 的列号
Unsatisfied = [3]

def inverse(data, *colNum):
    for colIndex in colNum:
        data[:,colIndex-1] = MinMaxScaler().fit_transform(data[:,colIndex-1])
        for i in range(data[:,colIndex-1].size):
            data[:, colIndex-1][i] = 1 - data[:, colIndex-1][i]

def z_score(data, *colNum):
    for colIndex in colNum:
        data[:,colIndex-1] = scale(data[:,colIndex-1])
        mapNormalDistributionToZ_score(data[:,colIndex-1])

def unsatisfied(data, *colNum):
    for colIndex in colNum:
        for i in range(data[:,colIndex-1].size):
            if data[:,colIndex-1][i] < 0:
                data[:, colIndex - 1][i] = -1
            elif data[:,colIndex-1][i] < 20:
                data[:, colIndex - 1][i] = 1 - data[:, colIndex - 1][i] * 0.05
            else:
                data[:, colIndex - 1][i] = 0

def mapNormalDistributionToZ_score(column):
    for num in range(column.size):
        if column[num] < -2:
            column[num] = 0.5
        elif column[num] < -1: # -2 <= x < -1，评分0.6
            column[num] = 0.6
        elif column[num] < 0: # -1 <= x < 0，评分0.7
            column[num] = 0.7
        elif column[num] < 1: # 0 <= x < 1，评分0.8
            column[num] = 0.8
        elif column[num] < 2: # 1 <= x < 2，评分0.9
            column[num] = 0.9
        else:         # x > 2，评分1
            column[num] = 1

def preprocessForKPI(data):
    z_score(data, *Z_score)
    unsatisfied(data, *Unsatisfied)
    inverse(data, *Inverse)

#数据预处理
f1 = open('renyuan.csv', 'r')
f1.readline()
reader = csv.reader(f1,delimiter=',')
my_matrix = np.ndarray((0,ColumnsInvolved+2))
for row in reader:
    tmp = np.array(row)
    my_matrix = np.vstack((my_matrix, tmp))

np.set_printoptions(suppress=True)
raw_data = my_matrix[:, 1:-1]
dataKPI = raw_data.copy() #KPI
dataKPI[dataKPI == ''] = 0.0
dataKPI = dataKPI.astype(np.float)
np.set_printoptions(precision=3)
preprocessForKPI(dataKPI)
print dataKPI
data = dataKPI.copy()
#minByCol, maxByCol = minMax(data)
sqrtWeightOfColumns = np.sqrt(WeightOfColumns)
data = data * sqrtWeightOfColumns


#计算服务厅评分
score = np.dot(dataKPI, WeightOfColumns)
#执行聚类，提取结果
clf = k_means(data,GroupsYouWant, n_init = 2000, return_n_iter = True, tol=1e-30)

centroids = clf[0]
results = clf[1]
inertia = clf[2]
iterations = clf[3]
centroids = centroids / sqrtWeightOfColumns

#按得到的聚类的分开存放
groupList = []
matrixInDf = pd.DataFrame(my_matrix)
groupNo = pd.DataFrame(results)
groupScore = pd.DataFrame(score)
concatWithGroupNo = pd.concat([matrixInDf, groupScore, groupNo], axis=1, ignore_index = True)
sortedByGroup = concatWithGroupNo.sort_values(by=ColumnsInvolved+3)
x =  sortedByGroup.as_matrix()[:,1:-3] #此处无用，为画图做准备
for i in range(GroupsYouWant):
    group = sortedByGroup[sortedByGroup[ColumnsInvolved+3] == i]
    groupList.append(group.as_matrix()[:,:-1])
'''
#降维，画图
model = Isomap(n_components=3, n_neighbors= len(data)-1)
#model = LocallyLinearEmbedding(n_components=3, n_neighbors= len(data)-1)
#model = SpectralEmbedding(n_components=3)
fitted = model.fit_transform(x)
figur = plt.figure("No name")
ax = Axes3D(figur)
plt.plot(fitted[:,0], fitted[:,1],fitted[:,2], 'ko') #黑色，和聚类后的点重叠了所以看不到
drawingParams = ['ro', 'yo', 'bo','go','mo','co','wo'] #红绿蓝黄紫青白
init = 0
count = 0
for group in groupList:
    iter = group.size/(ColumnsInvolved+2)
    f = fitted[init:iter+init]
    plt.plot(f[:, 0], f[:, 1], f[:, 2], drawingParams[count%7])
    init += iter
    count += 1
plt.show()
'''

#计算聚类中每一列的最大值，最小值
minArray = []
maxArray = []
for group in groupList:
    minInCol = [100000 for x in range(ColumnsInvolved)]
    maxInCol = [0 for x in range(ColumnsInvolved)]
    for row in group:
        count = 0
        for x in row[1:-2]:
            if float(x) > maxInCol[count]:
                maxInCol[count] = float(x)
            if float(x) < minInCol[count]:
                minInCol[count] = float(x)
            count += 1
    minArray.append(minInCol)
    maxArray.append(maxInCol)

#写到文件
f = open('result_广州人员_'+ str(GroupsYouWant)+'类_加权.csv', 'wb')
f1.seek(0)
header = f1.readline()
f.write(header)
writer = csv.writer(f)
count = 0
for group in groupList:
    f.write('Group '+str(count+1)+'\n')
    writer.writerows(group)
    f.write("Min")
    for num in minArray[count]:
        f.write(','+str(num))
    f.write("\n")
    f.write("Max")
    for num in maxArray[count]:
        f.write(','+str(num))
    f.write("\n")
    f.write("Centroid"+',')
    VIstring = ''.join(['%.2f,' % num for num in centroids[count]])
    VIstring = VIstring[:-1]
    f.write(VIstring)
    f.write("\n")
    f.write("Total,%d\n" % (group.size/(ColumnsInvolved + 2)))
    count += 1
    f.write("\n")
f.write("iterations:,%d, inertia:,%f" % (iterations, inertia))
f.write("\n")
f.write("weight:")
VIstring = ''.join(['%.2f,' % num for num in WeightOfColumns])
f.write(VIstring)
f.write("\n")
f.write("逐项评分后的数据\n")
writer.writerows(dataKPI)
f.close()
f1.close()



