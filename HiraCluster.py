import numpy as np
from Cluster import AgglomerativeClustering # Make sure to use the new one!!!
import sklearn
import sklearn.metrics
d = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)

clustering = AgglomerativeClustering(n_clusters=2, compute_full_tree=True,affinity='euclidean', linkage='complete')
clustering.fit(d)
print clustering.affinity
print clustering.distance


import numpy as np
from scipy.cluster.hierarchy import linkage

d1 = [
		[  0, 662, 877, 255, 412, 996],
		[662,   0, 295, 468, 268, 400],
		[877, 295,   0, 754, 564, 138],
		[255, 468, 754,   0, 219, 869],
		[412, 268, 564, 219,   0, 669],
		[996, 400, 138, 869, 669,   0]]
leaf_count = 3

X1 = [
		[  0, 662, 877, 255, 412, 996],
		[662,   0, 295, 468, 268, 400],
		[877, 295,   0, 754, 564, 138],
		[255, 468, 754,   0, 219, 869],
		[412, 268, 564, 219,   0, 669],
		[996, 400, 138, 869, 669,   0]]

leaf_labels=["0", "1", "2"]

#d = np.array(
#        [
#            [1, 2, 3],
##            [4, 5, 6],
#            [7, 8, 9]
#        ]
#)
print linkage(d, 'complete')



class Cluster:
	def __init__(self):
		pass
	def __repr__(self):
		return '(%s,%s)' % (self.left, self.right)
	def add(self, clusters, grid, lefti, righti):
		self.left = clusters[lefti]
		self.right = clusters[righti]
		# merge columns grid[row][righti] and row grid[righti] into corresponding lefti
		for r in grid:
			r[lefti] = min(r[lefti], r.pop(righti))
		grid[lefti] = map(min, zip(grid[lefti], grid.pop(righti)))
		clusters.pop(righti)
		return (clusters, grid)

def agglomerate(labels, grid):
	"""
	given a list of labels and a 2-D grid of distances, iteratively agglomerate
	hierarchical Cluster
	"""
	clusters = labels
	while len(clusters) > 1:
		# find 2 closest clusters
		print clusters
		distances = [(1, 0, grid[1][0])]
		for i,row in enumerate(grid[2:]):
			distances += [(i+2, j, c) for j,c in enumerate(row[:i+2])]
		j,i,_ = min(distances, key=lambda x:x[2])
		# merge i<-j
		c = Cluster()
		clusters, grid = c.add(clusters, grid, i, j)
		clusters[i] = c
	return clusters.pop()

if __name__ == '__main__':

	# Ref #1
	ItalyCities = ['BA','FI','MI','NA','RM','TO']
	ItalyDistances = [
		[  0, 662, 877, 255, 412, 996],
		[662,   0, 295, 468, 268, 400],
		[877, 295,   0, 754, 564, 138],
		[255, 468, 754,   0, 219, 869],
		[412, 268, 564, 219,   0, 669],
		[996, 400, 138, 869, 669,   0]]
print agglomerate(ItalyCities, ItalyDistances)


Z= []
node_dict = {}
n_samples = len(d)
def get_all_children(k, verbose=False):
    i,j = agg_cluster.children_[k]

    if k in node_dict:
        return node_dict[k]['children']

    if i < leaf_count:
        left = [i]
    else:
        # read the AgglomerativeClustering doc. to see why I select i-n_samples
        left = get_all_children(i-n_samples)

    if j < leaf_count:
        right = [j]
    else:
        right = get_all_children(j-n_samples)

    if verbose:
        print k,i,j,left, right
    left_pos = np.mean(map(lambda ii: d[ii], left),axis=0)
    right_pos = np.mean(map(lambda ii: d[ii], right),axis=0)

    # this assumes that agg_cluster used euclidean distances
    dist = sklearn.metrics.pairwise_distances([left_pos,right_pos],metric='euclidean')[0,1]

    all_children = [x for y in [left,right] for x in y]
    pos = np.mean(map(lambda ii: d[ii], all_children),axis=0)

    # store the results to speed up any additional or recursive evaluations
    node_dict[k] = {'top_child':[i,j],'children':all_children, 'pos':pos,'dist':dist, 'node_i':k + n_samples}
    return all_children
    #return node_di|ct

agg_cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=6)
agg_labels = agg_cluster.fit_predict(d)


for k,x in enumerate(agg_cluster.children_):
    get_all_children(k,verbose=False)

# Every row in the linkage matrix has the format [idx1, idx2, distance, sample_count].
Z = [[v['top_child'][0],v['top_child'][1],v['dist'],len(v['children'])] for k,v in node_dict.iteritems()]
# create a version with log scaled distances for easier visualization
Z_log =[[v['top_child'][0],v['top_child'][1],np.log(1.0+v['dist']),len(v['children'])] for k,v in node_dict.iteritems()]

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
plt.figure()
dn = hierarchy.dendrogram(Z_log,p=4,truncate_mode='level')
plt.show()







import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt


mat = np.array([
[0, 662, 877, 255, 412, 996],
[662, 0, 295, 468, 268, 400],
[877, 295, 0, 754, 564, 138],
[255, 468, 754, 0, 219, 869],
[412, 268, 564, 219, 0, 669],
[996, 400, 138, 869, 669, 0]]
)
dists = squareform(mat)
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels=["0", "1", "2","3","4","5"])
plt.title("test")
plt.show()



from scipy.cluster.hierarchy import dendrogram, linkage

data = [[0., 0.], [0.1, -0.1], [1., 1.], [1.1, 1.1]]

Z = linkage(data)

dendrogram(Z)
plt.title("dd")
plt.show()


from ete3 import Tree, TextFace, AttrFace
tree = Tree("((A, B), C);")
nodeA = tree & ("A")
nodeAB = tree.get_common_ancestor("A", "B")
nodeA.add_face(TextFace("mouse"), column=1, position="branch-right")
nodeA.add_face(TextFace("/human"), column=2, position="branch-right")
nodeAB.add_face(TextFace("mouse"), column=0, position="branch-top")
nodeAB.add_face(TextFace("/mouse"), column=1, position="branch-top")
nodeAB.add_face(AttrFace("dist", text_prefix="/"), column=2, position="branch-top")
tree.render("tree.png")
tree.show()

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import ete3

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
        branch0,branch0samples = go_down_tree(children,n_leaves,X,leaf_labels,node_children[0])
        branch1,branch1samples = go_down_tree(children,n_leaves,X,leaf_labels,node_children[1])
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

clusterer = AgglomerativeClustering(n_clusters=6,compute_full_tree=True) # You can set compute_full_tree to 'auto', but I left it this way to get the entire tree plotted
clusterer.fit(X) # X for whatever you want to fit
spanner = get_cluster_spanner(clusterer)
newick_tree = build_Newick_tree(clusterer.children_,clusterer.n_leaves_,X,leaf_labels,spanner) # leaf_labels is a list of labels for each entry in X
tree = ete3.Tree(newick_tree)
tree.show()