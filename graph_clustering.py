import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
def main():
    '''
    i=-1
    wf1 = open('data_avgRGB_1.csv','w',encoding='UTF8')
    wfw1 = csv.writer(wf1)
    f = open('dataskit4.csv','r',encoding='UTF8')
    rdr = csv.reader(f)
    for line in rdr:
        if(i%3 == 0):
            wfw1.writerow(line)
        i = i+1
        continue
    '''
    plt.style.use('seaborn')
    sns.set_palette("hls")
    data_Read = pd.read_csv('data_avgRGB_3.csv')
    print(data_Read.info())
    agg_clustering = AgglomerativeClustering(n_clusters = 2, linkage = 'average')
    read_one = data_Read.iloc[:,[1,2,3]]
    labels = agg_clustering.fit_predict(read_one)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection = '3d')
    x = data_Read['R']
    y = data_Read['G']
    z = data_Read['B']
    fd = data_Read['FileDir']
    new_csvgraphRGB_1 = open("3_avgRGB_rgb_average_cluster.csv","w",encoding = 'UTF8')
    csv_writer1 = csv.writer(new_csvgraphRGB_1)
    title = ['FileDir','R','G','B','labels']
    csv_writer1.writerow(title)
    for i  in range(len(x)):
        eachrow = []
        eachrow.append(fd[i])
        eachrow.append(x[i])
        eachrow.append(y[i])
        eachrow.append(z[i])
        eachrow.append(labels[i])
        csv_writer1.writerow(eachrow)
    ax.scatter(x,y,z,c = labels, s = 20, alpha = 0.5, cmap = 'rainbow')
    plt.show()

#picture 1,2,3
#label 
if __name__ == '__main__':
    main()