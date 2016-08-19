


import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np

def data_read(dir):
    file = open(dir)
    file.readline()
    reader = csv.reader(file)
    train = []
    for item in reader:
        tmp = []
        #tmp.append(int(item[0]))
        #for i in range(8,21):
        #    tmp.append(float(item[i]))
        tmp.append(float(item[10]))
        tmp.append(float(item[11]))
        tmp.append(float(item[12]))
        tmp.append(float(item[13]))
        tmp.append(float(item[14]))
        tmp.append(float(item[20]))
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
        now = item[5][:5]
        if item[5][:5] !=server:
            continue
        tmp.append(float(item[10]))
        tmp.append(float(item[11]))
        tmp.append(float(item[12]))
        tmp.append(float(item[13]))
        tmp.append(float(item[14]))
        tmp.append(float(item[20]))
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
            tmp.append(item[5])
            tmp.append(item[10])
            tmp.append(item[11])
            tmp.append(item[12])
            tmp.append(item[13])
            tmp.append(item[14])
            tmp.append(item[20])
            train.append(tmp)
            continue
        if item[5][:5] != server:
            continue
        tmp.append(float(item[5]))
        tmp.append(float(item[10]))
        tmp.append(float(item[11]))
        tmp.append(float(item[12]))
        tmp.append(float(item[13]))
        tmp.append(float(item[14]))
        tmp.append(float(item[20]))
        train.append(tmp)
    file.close()
    return train

max_min_scaler = preprocessing.MinMaxScaler()


#data = data_read("t_bsfwt.csv")
data = data_read_filter("t_bsfwt.csv", "24419")
k = 4 #group_num

#pca
pca = PCA(n_components=k).fit(data)
reduced_data = PCA(n_components=k).fit_transform(data)
clt= KMeans(init='k-means++', n_clusters=k, n_init=10)
clt1 = KMeans(init=pca.components_, n_clusters=k, n_init=1)

#scale
data_scale = preprocessing.scale(data)
#max_min
data_max_min = max_min_scaler.fit_transform(data)

clt.fit(data_max_min)

min_x = []
max_x = []

sum_num = [0,0,0,0,0,0,0,0]

#center = clt.cluster_centers_
center = max_min_scaler.inverse_transform(clt.cluster_centers_)
for i in range(0, k):
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

writer = csv.writer(open("result_fanwei_dongguan.csv","wb"))
for i in range(0,k):
    writer.writerow(min_x[i]+[sum_num[i]])
    writer.writerow(max_x[i])

for item in center:
    writer.writerow(item)

writer1 = csv.writer(open("dongguan.csv","wb"))
data_out = data_read1("t_bsfwt.csv", "24419")
for i in range(0,len(data_out)):
    now_line = data_out[i]
    if i==0:
        now_line.append("label")
        writer1.writerow(now_line)
        continue
    now_line.append(clt.labels_[i-1])
    writer1.writerow(now_line)