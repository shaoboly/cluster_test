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
#��ά
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
'''
#��Ҫ�۳ɼ��ࣿ���ǵ����ӻ�Ӧ<8
GroupsYouWant = 5
#�����ָ����������������е���������Ҫ������Ȩ�صĸ���һ��
ColumnsInvolved = 7
#Ȩ�أ�ҵ����	�����˴�	���������	����ռ�ձ�	ƽ������ʱ��	ҵ�����Ч��	����ʱ��
WeightOfColumns = [0.364121667,0.170238008,0.107485731,0.103761228,0.091030417,0.091030417,0.072332533] #���ݲ�ת��
WeightOfColumns = [0.042978215,0.080374517,0.145101754,0.155178897,0.17688102,0.17688102,0.222604578] #����ת��
#WeightOfColumns = 1.0/7 * np.array([1,1,1,1,1,1,1]) #����Ȩ

#��KPIʱ��Ҫ��ת���кţ���1��ʼ��
Inverse = [5,7]
#����KPIʱ��Ҫz_score���к�
Z_score = [1,2,4,6]
#1-0.05 * N ���к�
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
        elif column[num] < -1: # -2 <= x < -1������0.6
            column[num] = 0.6
        elif column[num] < 0: # -1 <= x < 0������0.7
            column[num] = 0.7
        elif column[num] < 1: # 0 <= x < 1������0.8
            column[num] = 0.8
        elif column[num] < 2: # 1 <= x < 2������0.9
            column[num] = 0.9
        else:         # x > 2������1
            column[num] = 1

def preprocessForKPI(data):
    z_score(data, *Z_score)
    unsatisfied(data, *Unsatisfied)
    inverse(data, *Inverse)

#����Ԥ����
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


#�������������
score = np.dot(dataKPI, WeightOfColumns)
#ִ�о��࣬��ȡ���
clf = k_means(data,GroupsYouWant, n_init = 2000, return_n_iter = True, tol=1e-30)

centroids = clf[0]
results = clf[1]
inertia = clf[2]
iterations = clf[3]
centroids = centroids / sqrtWeightOfColumns

#���õ��ľ���ķֿ����
groupList = []
matrixInDf = pd.DataFrame(my_matrix)
groupNo = pd.DataFrame(results)
groupScore = pd.DataFrame(score)
concatWithGroupNo = pd.concat([matrixInDf, groupScore, groupNo], axis=1, ignore_index = True)
sortedByGroup = concatWithGroupNo.sort_values(by=ColumnsInvolved+3)
x =  sortedByGroup.as_matrix()[:,1:-3] #�˴����ã�Ϊ��ͼ��׼��
for i in range(GroupsYouWant):
    group = sortedByGroup[sortedByGroup[ColumnsInvolved+3] == i]
    groupList.append(group.as_matrix()[:,:-1])
'''
#��ά����ͼ
model = Isomap(n_components=3, n_neighbors= len(data)-1)
#model = LocallyLinearEmbedding(n_components=3, n_neighbors= len(data)-1)
#model = SpectralEmbedding(n_components=3)
fitted = model.fit_transform(x)
figur = plt.figure("No name")
ax = Axes3D(figur)
plt.plot(fitted[:,0], fitted[:,1],fitted[:,2], 'ko') #��ɫ���;����ĵ��ص������Կ�����
drawingParams = ['ro', 'yo', 'bo','go','mo','co','wo'] #�������������
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

#���������ÿһ�е����ֵ����Сֵ
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

#д���ļ�
f = open('result_������Ա_'+ str(GroupsYouWant)+'��_��Ȩ.csv', 'wb')
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
f.write("�������ֺ������\n")
writer.writerows(dataKPI)
f.close()
f1.close()



