#coding:GBK

import numpy as np
import csv
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from cc import scaleMaxToOne

import sklearn.neighbors.typedefs


import ctypes
import os
import sys

if getattr(sys, 'frozen', False):
    # Override dll search path.
    ctypes.windll.kernel32.SetDllDirectoryW('C:/Anaconda2/Library/bin')
    # Init code to load external dll
    ctypes.CDLL('mkl_avx2.dll')
    ctypes.CDLL('mkl_def.dll')
    ctypes.CDLL('mkl_vml_avx2.dll')
    ctypes.CDLL('mkl_vml_def.dll')

    # Restore dll search path.
    ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)


weight = [0.050417881,0.061460086,0.082719141,0.127127107,0.174003716,0.230284059,0.27398801] #广州转置

def data_read(dir):
    file = open(dir)
    file.readline()
    reader = csv.reader(file)
    train = []
    count = 0
    for item in reader:
        tmp = []
        #tmp.append(int(item[0]))
        #for i in range(8,21):
        #    tmp.append(float(item[i]))

        for i in range(2,9):
            tmp.append(float(item[i]))
        count+=1
        print count
        train.append(tmp)
    file.close()
    return np.array(train)

def data_read_filter(dir,server):
    file = open(dir)
    file.readline()
    reader = csv.reader(file)
    train  = []
    for item in reader:
        tmp = []
        #tmp.append(int(item[0]))
        #for i in range(8,21):
        #    tmp.append(float(item[i]))
        now = item[1][:5]
        if item[1][:5] !=server:
            continue
        for i in range(2, 9):
            tmp.append(float(item[i]))
        train.append(tmp)
    file.close()
    return np.array(train)

def data_read1(dir,server):
    file = open(dir)
    reader = csv.reader(file)
    train  = []
    count = 0
    for item in reader:
        tmp = []
        if count ==0:
            count+=1
            for i in range(0, 9):
                tmp.append(item[i])
            train.append(tmp)
            continue

        for i in range(0, 9):
            tmp.append(item[i])
        train.append(tmp)
    file.close()
    return train

def trans_zhong(X):
    for item in X:
        for i in range(0,7):
            item[i]*=math.sqrt(weight[i])

    return X
def trans_ni(X):
    for item in X:
        for i in range(0, 7):
            item[i] /= math.sqrt(weight[i])
    return X

max_min_scaler = preprocessing.MinMaxScaler()


data = data_read("t_bsfwt_stard.csv")
#data = data_read_filter("t_bsfwt_stard.csv", "24419")
group_num = 4 #类个数

#pca分析
#pca = PCA(n_components=group_num).fit(data)
#reduced_data = PCA(n_components=group_num).fit_transform(data)
clt= KMeans(init='k-means++', n_clusters=group_num, n_init=10)
#clt1 = KMeans(init=pca.components_, n_clusters=group_num, n_init=1)

print "clt_sucess"

#标准化
#data_scale = preprocessing.scale(data)
#归一化

maxx, minn = scaleMaxToOne(data)

data_max_min = max_min_scaler.fit_transform(data)
trans_zhong(data_max_min)
print "process"
clt.fit(data_max_min)
print "fit"
min_x = []
max_x = []

sum_num = [0,0,0,0,0,0,0,0]

#center = clt.cluster_centers_
trans_ni(clt.cluster_centers_)
center = max_min_scaler.inverse_transform(clt.cluster_centers_)
for i in range(0, group_num):
    tmp_mi = []
    tmp_ma = []
    for j in range(0,len(data[0])):
        tmp_mi.append(1000000.0)
        tmp_ma.append(0.0)
    min_x.append(tmp_mi)
    max_x.append(tmp_ma)

count = 0
for item in clt.labels_:
    for j in range(0, len(data[0])):
        if min_x[item][j]>data[count][j]:
            min_x[item][j] = data[count][j]
        if max_x[item][j]<data[count][j]:
            max_x[item][j] = data[count][j]
    sum_num[item]+=1

    count +=1

print min_x
print max_x
print sum_num
print clt.inertia_

data_out = data_read1("t_bsfwt_stard.csv", "24419")
writer = csv.writer(open("result_fanwei_standard.csv","wb"))

writer.writerow(['类']+data_out[0][2:9]+['sum'])

for i in range(0,group_num):

    writer.writerow([i]+min_x[i]+[sum_num[i]])
    writer.writerow([' ']+max_x[i])

for item in center:
    writer.writerow(item)

writer1 = csv.writer(open("standard.csv","wb"))

for i in range(0,len(data_out)):
    now_line = data_out[i]
    if i==0:
        now_line.append("类编号")
        writer1.writerow(now_line)
        continue
    now_line.append(clt.labels_[i-1])
    writer1.writerow(now_line)