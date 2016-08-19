# -*-coding:utf-8 -*-
import numpy as np
from sklearn.manifold import TSNE
import numpy
import sklearn
from pylab import *
from scipy.cluster.vq import *
from sklearn.cluster import k_means
from sklearn import preprocessing
import csv


def scaleMaxToOne(data):
    maxx = 0
    minn = 100000
    for rows in data:
        for numbers in rows:
            if numbers > maxx:
                maxx = numbers
            if numbers < minn:
                minn = numbers
    row = 0
    for rows in data:
        col = 0
        for numbers in rows:
            data[row][col] = (data[row][col] - minn) * 1.0 / (maxx - minn)
            col += 1
        row += 1
    return (maxx, minn)

if __name__ == '__main__':
    f1 = open('t_bsfwt.csv', 'r')
    my_matrix = numpy.loadtxt(f1,delimiter=",",skiprows=1)
    data = my_matrix[:, 1:].copy()

    maxx, minn = scaleMaxToOne(data)
    clf = k_means(data, 3, n_init=10, return_n_iter=True)
    centroids = clf[0]
    results = clf[1]
    inertia = clf[2]
    iterations = clf[3]
    centroids = centroids * (maxx-minn) + minn
    print centroids

    #为了分开存储每个聚类创建每个初始矩阵(应该不用那么麻烦才对)
    group1 = np.ones(9)
    group2 = np.ones(9)
    group3 = np.zeros(9)
    groupList = []
    groupList.append(group1)
    groupList.append(group2)
    groupList.append(group3)

    count = 0
    for num in results:
        groupList[num] =  numpy.vstack((groupList[num], my_matrix[count]))
        count += 1
    count = 0

    #去除创建时加入的第一行ones
    for group in groupList:
        groupList[count] = groupList[count][1:]
        count += 1


    #计算聚类中每一列的最大值，最小值
    minArray = []
    maxArray = []
    for group in groupList:
        minInCol = [100000 for x in range(8)]
        maxInCol = [0 for x in range(8)]
        for row in group:
            count = 0
            for x in row[1:]:
                if x > maxInCol[count]:
                    maxInCol[count] = x
                if x < minInCol[count]:
                    minInCol[count] = x
                count += 1
        minArray.append(minInCol)
        maxArray.append(maxInCol)

    f = open('result.csv', 'wb')
    f1.seek(0)
    header = f1.readline()
    f.write(header)
    writer = csv.writer(f)
    count = 0
    for group in groupList:
        f.write('Group'+str(count+1)+'\n')
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
        count += 1
        f.write("\n")




