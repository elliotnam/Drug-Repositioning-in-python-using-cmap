import numpy as np
from sklearn.cluster import AgglomerativeClustering
import ete3
import numpy as np
import sframe as sf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
import scipy.spatial.distance as dist
from sklearn import manifold

d= np.array([
		[  0, 662, 877, 255, 412, 996],
		[662,   0, 295, 468, 268, 400],
		[877, 295,   0, 754, 564, 138],
		[255, 468, 754,   0, 219, 869],
		[412, 268, 564, 219,   0, 669],
		[996, 400, 138, 869, 669,   0]]

)

X = np.array([
		[  0, 662, 877, 255, 412, 996],
		[662,   0, 295, 468, 268, 400],
		[877, 295,   0, 754, 564, 138],
		[255, 468, 754,   0, 219, 869],
		[412, 268, 564, 219,   0, 669],
		[996, 400, 138, 869, 669,   0]]

)
leaf_labels=["A", "B", "C","D","E","F"]

rankInfo = pd.read_csv("/home/elliotnam/project/cmap/fingerprintresult.csv",header=None)

ss = np.arange(1922)

ss = ss.astype(str)
print(ss)
#print(rankInfo[[119,1]])
print(".............")
print rankInfo.iloc[119,0]
rankInfo[119][0] = "HNMPA-AM3"
#print(rankInfo[119][0])
#print(rankInfo['acetylsalicylic acid'])
tt =  rankInfo.values
#kk = tt[1:200,1:200]
kk = tt[1:,1:]
kk = kk.astype(float)
print(kk.shape)
jj = tt[0]

#leaf_labels = jj[1:200]
#leaf_labels = jj[1:250]
#leaf_labels = ss
leaf_labels=jj
print(leaf_labels.shape)

X= kk

def build_Newick_tree(children,n_leaves,X,leaf_labels,spanner):
    """
    build_Newick_tree(children,n_leaves,X,leaf_labels,spanner)

    Get a string representation (Newick tree) from the sklearn
    AgglomerativeClustering.fit output.

    Input:
        children: AgglomerativeClustering.children_
        n_leaves: AgglomerativeClustering.n_leaves_
        X: parameters supplied to AgglomerativeClustering.fit
        leaf_labels: The label of each parameter array in X
        spanner: Callable that computes the dendrite's span

    Output:
        ntree: A str with the Newick tree representation

    """
    return go_down_tree(children,n_leaves,X,leaf_labels,len(children)+n_leaves-1,spanner)[0]+';'

def go_down_tree(children,n_leaves,X,leaf_labels,nodename,spanner):
    """
    go_down_tree(children,n_leaves,X,leaf_labels,nodename,spanner)

    Iterative function that traverses the subtree that descends from
    nodename and returns the Newick representation of the subtree.

    Input:
        children: AgglomerativeClustering.children_
        n_leaves: AgglomerativeClustering.n_leaves_
        X: parameters supplied to AgglomerativeClustering.fit
        leaf_labels: The label of each parameter array in X
        nodename: An int that is the intermediate node name whos
            children are located in children[nodename-n_leaves].
        spanner: Callable that computes the dendrite's span

    Output:
        ntree: A str with the Newick tree representation

    """
    nodeindex = nodename-n_leaves
    if nodename<n_leaves:
        return leaf_labels[nodeindex],np.array([X[nodeindex]])
    else:
        node_children = children[nodeindex]
        branch0,branch0samples = go_down_tree(children,n_leaves,X,leaf_labels,node_children[0],spanner)
        branch1,branch1samples = go_down_tree(children,n_leaves,X,leaf_labels,node_children[1],spanner)
        node = np.vstack((branch0samples,branch1samples))
        branch0span = spanner(branch0samples)
        branch1span = spanner(branch1samples)
        nodespan = spanner(node)
        branch0distance = nodespan-branch0span
        branch1distance = nodespan-branch1span
        nodename = '({branch0}:{branch0distance},{branch1}:{branch1distance})'.format(branch0=branch0,branch0distance=branch0distance,branch1=branch1,branch1distance=branch1distance)
        return nodename,node

def get_cluster_spanner(aggClusterer):
    """
    spanner = get_cluster_spanner(aggClusterer)

    Input:
        aggClusterer: sklearn.cluster.AgglomerativeClustering instance

    Get a callable that computes a given cluster's span. To compute
    a cluster's span, call spanner(cluster)

    The cluster must be a 2D numpy array, where the axis=0 holds
    separate cluster members and the axis=1 holds the different
    variables.

    """
    if aggClusterer.linkage=='ward':
        if aggClusterer.affinity=='euclidean':
            spanner = lambda x:np.sum((x-aggClusterer.pooling_func(x,axis=0))**2)
    elif aggClusterer.linkage=='complete':
        if aggClusterer.affinity=='euclidean':
            spanner = lambda x:np.max(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
        elif aggClusterer.affinity=='l1' or aggClusterer.affinity=='manhattan':
            spanner = lambda x:np.max(np.sum(np.abs(x[:,None,:]-x[None,:,:]),axis=2))
        elif aggClusterer.affinity=='l2':
            spanner = lambda x:np.max(np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)))
        elif aggClusterer.affinity=='cosine':
            spanner = lambda x:np.max(np.sum((x[:,None,:]*x[None,:,:]))/(np.sqrt(np.sum(x[:,None,:]*x[:,None,:],axis=2,keepdims=True))*np.sqrt(np.sum(x[None,:,:]*x[None,:,:],axis=2,keepdims=True))))
        else:
            raise AttributeError('Unknown affinity attribute value {0}.'.format(aggClusterer.affinity))
    elif aggClusterer.linkage=='average':
        if aggClusterer.affinity=='euclidean':
            spanner = lambda x:np.mean(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
        elif aggClusterer.affinity=='l1' or aggClusterer.affinity=='manhattan':
            spanner = lambda x:np.mean(np.sum(np.abs(x[:,None,:]-x[None,:,:]),axis=2))
        elif aggClusterer.affinity=='l2':
            spanner = lambda x:np.mean(np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)))
        elif aggClusterer.affinity=='cosine':
            spanner = lambda x:np.mean(np.sum((x[:,None,:]*x[None,:,:]))/(np.sqrt(np.sum(x[:,None,:]*x[:,None,:],axis=2,keepdims=True))*np.sqrt(np.sum(x[None,:,:]*x[None,:,:],axis=2,keepdims=True))))
        else:
            raise AttributeError('Unknown affinity attribute value {0}.'.format(aggClusterer.affinity))
    else:
        raise AttributeError('Unknown linkage attribute value {0}.'.format(aggClusterer.linkage))
    return spanner

clusterer = AgglomerativeClustering(n_clusters=10,compute_full_tree=True,linkage='ward',affinity='euclidean') # You can set compute_full_tree to 'auto', but I left it this way to get the entire tree plotted
clusterer.fit(X) # X for whatever you want to fit

print(clusterer.labels_)

#print({i: np.where(clusterer.labels_ == i)[0] for i in range(clusterer.n_clusters)})

for i in range(clusterer.n_clusters):
    print(i)
    kk = np.where(clusterer.labels_ == i)[0]
    print(kk.size)
    print(kk)

#print({i: X[np.where(clusterer.labels_ == i)] for i in range(clusterer.n_clusters)})
spanner = get_cluster_spanner(clusterer)
print(clusterer.n_leaves_)
print(clusterer.labels_)
newick_tree = build_Newick_tree(clusterer.children_,clusterer.n_leaves_,X,leaf_labels,spanner) # leaf_labels is a list of labels for each entry in X
tree = ete3.Tree(newick_tree)
tree.show()