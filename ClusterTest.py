import numpy as np
import sframe as sf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
import scipy.spatial.distance as dist
from sklearn import manifold
#from sklearn.cluster import AgglomerativeClustering

#np.random.seed(4711)  # for repeatability of this tutorial
#a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
#b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
#X = np.concatenate((a, b),)
#print(X)
#print X.shape  # 150 samples with 2 dimensions

rankInfo = pd.read_csv("/home/elliotnam/project/cmap/fingerprintresult.csv",header=None)

tt =  rankInfo.values
kk = tt[1:,1:]
kk = kk.astype(float)
print(kk.shape)


jj = tt[0]

y = jj[1:]

from sklearn.cluster import AgglomerativeClustering
from time import time





def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

model = AgglomerativeClustering(linkage='ward',n_clusters=10,affinity='euclidean')
X_red = manifold.SpectralEmbedding(n_components=10).fit_transform(kk)
t0 = time()
ss = model.fit(kk)

#plt.figure()
#plt.axes([0, 0, 1, 1])
##for l, c in zip(np.arange(model.n_clusters), 'rgbk'):#
   # plt.plot(kk[model.labels_ == l].T, c=c, alpha=.5)
   # plt.axis('tight')
   # plt.axis('off')
   # plt.suptitle("AgglomerativeClustering(affinity=%s)" % 'euclidean', size=20)


#plt.show()




plot_clustering(X_red, kk, model.labels_, "%s linkage" % 'ward')









# plot_clustering(kk,)
linkageMatrix = linkage(kk,'ward')

distMatrix = dist.pdist(kk)
c,coph_dists = cophenet()
#distSquareMatrix = dist.squareform(distMatrix)
linkageMatrix = linkage(distMatrix,'ward')

dendro = dendrogram(linkageMatrix)
leaves = dendro['leaves']
transformedData = kk[leaves,:]

print(rankInfo.dtypes)
#print(rankInfo.column_names())
#plt.scatter(rankInfo[:,0], rankInfo[:,1])
#plt.show()
#plt.interactive(False)
#rankInfo2 = rankInfo.cumsum()
#rankInfo2.plot.scatter(x='metformin',y='metformin',color="DarkGreen")
#rankInfo.plot.area()
#rankInfo.plot.barh(stacked=True)
#plt.show()

Z = linkage(kk,'ward')
print(Z)

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c,copy_dist = cophenet(Z,pdist(kk))

print(c)

print(Z[0])

idxs = [33, 68, 62]
plt.figure(figsize=(10, 8))
plt.scatter(kk[:,0], kk[:,1])  # plot all points
plt.scatter(kk[idxs,0], kk[idxs,1], c='r')  # plot interesting points in red again
plt.show()