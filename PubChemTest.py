import pubchempy as pcp
import pybel2 as pb
from pubchempy import Compound,get_compounds
import pandas as pd
import numpy as np
import openbabel
import scipy
from scipy.spatial import distance
import gc
from sklearn.cluster import AgglomerativeClustering
from rdkit import Chem
from rdkit.Chem import MCS
import cPickle
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



def showHiraClusterTree(clusterPath):
    dataHolder = pd.read_csv("/home/elliotnam/project/cmap/genedistanceresult_id_column.csv")
    tt = dataHolder.values
    X = tt[0:, 0:].astype(float)

    #kk = dataHolder.iloc[startPoint:,startPoint:].astype(float)
    leaf_labels = dataHolder.columns[0:]
    with open(clusterPath,"rb") as fid:
            ccluster = cPickle.load(fid)

    spanner = get_cluster_spanner(ccluster)
    print(ccluster.n_leaves_)
    print(ccluster.labels_)
    newick_tree = build_Newick_tree(ccluster.children_, ccluster.n_leaves_, X, leaf_labels,
                                    spanner)  # leaf_labels is a list of labels for each entry in X
    tree = ete3.Tree(newick_tree)
    tree.show()
    '''
    for i in range(ccluster.n_clusters):
        print(i)
        kk = np.where(ccluster.labels_ == i)[0]
        print(kk.size)
        print(kk)
    '''

def testPubChem():
    dName = "BCB000040"
    dResult = get_compounds(dName, 'name')
    if (dResult.__len__() != 0):
        print(dResult[0].cid)
        ddd = dResult[0].isomeric_smiles  # [dName, dResult[0].isomeric_smiles]
        print(ddd)

def getSmilesFromPubChem():

    dataResult = pd.DataFrame(columns=('instance_id','drugName','smiles'))

    drugNames = pd.read_excel("/home/elliotnam/project/cmap/cmap_instances_02.xls",sheetname="build02")

    dataResult.instance_id = drugNames.instance_id
    dataResult.drugName = drugNames.cmap_name
    #dataResult.index = drugNames.instance_id
    #dataResult.smiles[0] = "tt"
    dataResult.dropna(subset=['drugName'],inplace=True)
    print(dataResult.drugName)
    #drugNames2 = drugNames.cmap_name.sort_values().unique()
    print(drugNames.shape)
    i = 0

    for dName in  dataResult.drugName:
        print("...drug name..........")
        print(dName)
        print(i)
        #dataResult.index = dataResult.instance_id
        if pd.isnull(dName) == False:
            dResult = get_compounds(dName,'name')
            if(dResult.__len__() != 0):
                print(dResult[0].cid)
                dataResult.smiles[i] = dResult[0].isomeric_smiles #[dName, dResult[0].isomeric_smiles]
                print(dataResult.smiles[i])
                i += 1
            '''
            for cs in dResult:testPubChem
                print(i)
                print(cs.cid)
                #print(cs.get_cid())
                dataResult.loc[i] =[dName,cs.isomeric_smiles]
            '''
        else :
            print("now null ")
            print(i)
    dataResult.dropna(subset=['smiles'], inplace=True)
    print(dataResult)
    #dataResult.index = dataResult.instance_id
    dataResult.to_csv("/home/elliotnam/project/cmap/drug_smiles.csv",index=False)


def cleanDrugMiles():
    dataResult = pd.read_csv("/home/elliotnam/project/cmap/drug_smiles.csv")
    dataResult.dropna(subset=['smiles'], inplace=True)
    print(dataResult)
    dataResult.to_csv("/home/elliotnam/project/cmap/drug_smiles2.csv")

def calcGeneDistances():

    rankInfo = pd.read_csv("/home/elliotnam/project/cmap/rankMatrix.csv",sep="\t")
    print(rankInfo.shape[1])
    print(rankInfo.shape)
    colNames = rankInfo.columns
    tt = rankInfo.values
    #print(tt[0,6101])
    print(rankInfo.iloc[0:,6100])
    print(rankInfo.iloc[0:,1])
    #print(rankInfo.tail())
    #print(rankInfo.iloc[0:,1:3])
    lsize = rankInfo.shape[1]-2
    print(lsize)
    holder = np.zeros((lsize,lsize))
    print(holder.shape)
    '''
    for index1,row1 in rankInfo.iterrows():
        for index2,row2 in rankInfo.iterrows():
            holder[index1,index2] = distance.euclidean(row1[1:-2], row2[1:-2])
            print(index1,index2)
pd.read_csv("/home/elliotnam/project/cmap/genedistanceresult_id_column.csv")
    print(holder)

    calculate rows X rows
    '''

    for i in range(1,lsize+1):  #It is positon not size
        for j in range(1,lsize+1):
            s1 = rankInfo.iloc[0:,i]
            s2 = rankInfo.iloc[0:,j]
            holder[i-1,j-1] = distance.euclidean(rankInfo.iloc[0:,i],rankInfo.iloc[0:,j])
            print(i-1,j-1)
    print(holder)
    np.savetxt("/home/elliotnam/project/cmap/genedistanceresult_id.csv", holder, delimiter=",")

def makeDrugFingerPrintData():
    sData = pd.read_csv("/home/elliotnam/project/cmap/drug_smiles.csv")


    #ddName = sData.drugName.sort_values().unique()
    #ddName = sData.drugName.unique()
    ddName = sData.drugName
    fdaData = pd.read_csv("/home/elliotnam/project/cmap/druglinks.csv")

    fdaData =  fdaData.apply(lambda x: x.astype(str).str.lower())

    fdaFlag = np.zeros((len(ddName),3))
    smiles = []

    i = 0
    for dName in ddName:
        #if(fdaData.Name.str.isin(dName).any(1)):
        instID = sData[sData['drugName']==dName]
        #print(instID.iloc[0].instance_id)
        dd = fdaData[fdaData['Name'] == dName]
        #print(dName)
        #print(dd.__len__())
        if(dd.__len__() != 0):
            #print("it is on")
            fdaFlag[i,1] =1
        else :
            fdaFlag[i, 1] = 0
        #dd = fdaData[fdaData['Name'].isin([dName])]
        fdaFlag[i, 0] = instID.iloc[0].instance_id
        smiles.append(sData.iloc[i].smiles)
        i+=1


    d = {'instance_id':fdaFlag[:,0],'Name':ddName,'smiles':smiles,'isApproved':fdaFlag[:,1]}
    df = pd.DataFrame(data=d)

    df =saveFTFromSmiles(df)
    df.to_csv("/home/elliotnam/project/cmap/DrugFingerPrint.csv", index=False)
    #df = saveGeneInfo(df)
    #df.to_csv("/home/elliotnam/project/cmap/DrugFingerPrint.csv",index=False)


def saveGeneInfo(fileName):
    rankInfo = pd.read_csv("/home/elliotnam/project/cmap/rankMatrix.csv", sep="\t")
    dataHolder = pd.read_csv("/home/elliotnam/project/cmap/DrugFingerPrint.csv")
    print(len(dataHolder[dataHolder['isApproved'] == 1]))
    print(len(dataHolder[dataHolder['isApproved'] == 0]))
    print(len(dataHolder))
    #print(rankInfo.columns)
    print(rankInfo.index)

    df = []
    i =0

    print(len(dataHolder['instance_id']))
    for ss in dataHolder['instance_id']:
        if i == 0:
            dd = rankInfo[[str(int(ss))]].values.T
            df = pd.DataFrame(dd.flatten()).T
            df.columns = rankInfo["probe_id"]

            #df.append(pd.DataFrame(dd,columns=[rankInfo.probe_id]))
            #print(df)

        #dd = ",".join(str(x) for x in holder)checkIsFdaApproved
        else:
            #df2 = pd.DataFrame(data=holder)
            print("..........................")
            print(ss)
            dd = rankInfo[[str(int(ss))]].T.values
            #print(dd.shape)
            df.loc[len(df)] = dd.flatten()
        i += 1
        print(i)
        #print(df)
    #print(df)
    dataHolder = pd.concat([dataHolder, df], axis=1)
    dataHolder.to_csv("/home/elliotnam/project/cmap/DrugFingerPrint_final.csv",index=False)
    #return dataHolder

    #print(df.shape)


def checkIsFdaApproved():
    sData = pd.read_csv("/home/elliotnam/project/cmap/drug_smiles.csv")
    print(sData.shape).probe_id
    ddName = sData['drugName']
    fdaFlag = np.zeros(sData.shape[0])
    fdaData = pd.read_csv("/home/elliotnam/project/cmap/druglinks.csv")
    fName = fdaData['Name']

    print(fdaData['Name'])

    d= fdaData[fdaData['Name']== "metformin"]
    print("test")

    for idx,fn in ddName.iteritems():
        if fdaData['Name'].str.contains(fn):
            fdaFlag[idx] =1

    print(fdaFlag)
    sData['FDAApproved'] = np.where(fName.str.contains(fName))

def saveFTFromSmiles(dataHolder):
    #columns=('instance_id', 'drugName', 'fingerPrint'))

    dd = dataHolder["smiles"]

    holder = np.zeros(166)

    xMols = [pb.readstring("smi", x) for x in dd]
    fps = [x.calcfp("MACCS") for x in xMols]
    print(fps)




    df = pd.DataFrame().T
    i = 0
    for xx in fps:
        jj = xx.bits
        holder[jj] = 1

        if i == 0:
            df = pd.DataFrame(holder).T
            print(df.shape)
            i=1
        #dd = ",".join(str(x) for x in holder)checkIsFdaApproved
        else:
            #df2 = pd.DataFrame(data=holder)
            df.loc[len(df)] = holder

        #df2 = pd.DataFrame(data=holder)
        #df = pd.concat([df, df2], axis=1)
        holder[:] = 0
        #print(df)

    dataHolder = pd.concat([dataHolder,df],axis=1)
    print(dataHolder)
    return dataHolder #dataHolder.to_csv("/home/elliotnam/project/cmap/DrugFingerPrint.csv",index=False)

# np.savetxt("/home/elliotnam/project/cmap/drug_fingerprint.csv", fps, delimiter=",",fmt='%s')


def calcFPFromSmiles():

    sData = pd.read_csv("/home/elliotnam/project/cmap/drug_smiles.csv")

    dd = sData["smiles"]
    holder = np.zeros((dd.size,dd.size))


    xMols = [pb.readstring("smi",x) for x in dd]
    fps = [x.calcfp("MACCS") for x in xMols]
    print(fps)


    for xx in fps:
        jj = xx.__str__()
        jj = map(int,jj.split(", "))
        print('xx:')
        print(jj.__len__())
        print(jj)

        jj = xx.bits
        holder[jj] = 1
        print(holder)

        print('bits len:')
        print(jj.__len__())
        print(jj)
    #np.savetxt("/home/elliotnam/project/cmap/drug_fingerprint.csv", fps, delimiter=",",fmt='%s')
    exit()
    i = 0
    for xfps in fps:
        j=0
        for yfps in fps:
            holder[i,j] = xfps | yfps   # | operator is overloaded to compute Tanimoto coefficient
            j+=1
        i+=1
        print(i,j)
    print(holder)
    #dResult = pd.DataFrame(holder,index=sData["instance_id"],columns=sData["instance_id"])
    dResult = pd.DataFrame(holder, index=sData["instance_id"], columns=sData["instance_id"])
    dResult.to_csv("/home/elliotnam/project/cmap/fingerprintresult.csv",index=False)


def clusteringInfos(no_cluster,dataHolder,startPoint):

    #rankInfo = pd.read_csv(dataPath)# "/home/elliotnam/project/cmap/fingerprintresult.csv", header=None)

    tt = dataHolder.values
    X = tt[0:, startPoint:].astype(float)

    #kk = dataHolder.iloc[startPoint:,startPoint:].astype(float)
    leaf_labels = dataHolder.columns[0:]

    print(leaf_labels.shape)

    cluster = AgglomerativeClustering(n_clusters=no_cluster, compute_full_tree=True, linkage='ward',
                                        affinity='euclidean')  # You can set compute_full_tree to 'auto', but I left it this way to get the entire tree plotted
    cluster.fit(X)  # X for whatever you want to fit



    #print(cluster.labels_)

    # print({i: np.where(clusterer.labels_ == i)[0] for i in range(clusterer.n_clusters)})
    return cluster

'''
    for i in range(cluster.n_clusters):
        print(i)
        kk = np.where(cluster.labels_ == i)[0]
        print(kk.size)
        print(kk)
'''


def getMCS(idxs):
    sData = pd.read_csv("/home/elliotnam/project/cmap/drug_smiles.csv")

    result = 0

    rList = []
    #print(sData[1:2])
    rHolder = sData.iloc[idxs]['smiles']
    print(sData.iloc[idxs]['smiles'])

    for  sString in rHolder:
        rList.append(Chem.MolFromSmiles(sString))

    res= MCS.FindMCS(rList)#dataHolder = pd.concat([dataHolder,df],axis=1)
    print(res)
    if(res.numAtoms > 0):
        result =1

    return result

def runGenModel(filePath):
    with open(filePath + ".pkl","rb") as fid:
            ccluster = cPickle.load(fid)

    #df = pd.DataFrame(columns=range(0,ccluster.n_clusters))
    df = pd.DataFrame()
    for i in range(ccluster.n_clusters):
        print(i)
        kk = np.where(ccluster.labels_ == i)[0]
        df2 = pd.DataFrame(data=kk)
        df =  pd.concat([df,df2],axis=1)

    df.columns = [range(0,ccluster.n_clusters)]
    df.to_csv(filePath + "_clustering_result.csv",index=False)


def runCluster(cluNo,dataList,path,startPoint):
    ccluster = clusteringInfos(cluNo, dataList, startPoint)

    with open(path,"wb") as fid:
        cPickle.dump(ccluster,fid)

    return ccluster

def runMCS(cluster):
    successNo = 0
    failNo = 0
    for i in range(cluster.n_clusters):
        print(i)
        kk = np.where(cluster.labels_ == i)[0]
            #print(kk.size)
            #print(kk)
        if(getMCS(kk) == 1):
            successNo+=1
        else :
            failNo+=1

    print("success:%d" % successNo)
    print("fail : %d" % failNo)



def modifyDrugData():
    rankInfo = pd.read_csv("/home/elliotnam/project/cmap/genedistanceresult_id.csv", header=None)
    drugNames = pd.read_excel("/home/elliotnam/project/cmap/cmap_instances_02.xls",sheetname="build02")

    drugNames2 = drugNames.cmap_name.sort_values().unique()


def seeGenData():
    #tt = np.genfromtxt("/home/elliotnam/project/cmap/genedistanceresult_id.csv",delimiter=',', dtype=None)
    rankInfo = pd.read_csv("/home/elliotnam/project/cmap/genedistanceresult_id.csv",header=None)

    rankInfo2 = pd.read_csv("/home/elliotnam/project/cmap/rankMatrix.csv", sep="\t")
    colNames = rankInfo2.columns
    print(rankInfo.shape)
    print(rankInfo2.columns)
    jj = colNames[1:-1]
    rankInfo.columns = jj
    print(rankInfo)
    #print(rankInfo.tail())
    print(rankInfo.iloc[:,6099])
    print(rankInfo.iloc[6099,:])
    print(rankInfo.iloc[0,:])
    print(rankInfo.iloc[:,3])
    rankInfo.to_csv("/home/elliotnam/project/cmap/genedistanceresult_id_column.csv",index=False)
    print('end')
    #rankInfo.drop(rankInfo.indrankInfo[[ss]].valuesex[6099],inplace=True)
    #rankInfo.drop(rankInfo.columns[6099],axis=1,inplace=True)
    #rankInfo.to_csv("/home/elliotnam/project/cmap/genedistanceresult_id.csv",header=False,index=False)
    #tt = rankInfo.values
#    print(tt)
    #print("")


def compareClusters():
    genCluster = pd.read_csv("/home/elliotnam/project/cmap/genedistanceresult_id_column_clustering_result.csv")
    chemCluster = pd.read_csv("/home/elliotnam/project/cmap/fingerprintresult_clustering_result.csv")

    resultNo = np.zeros((len(genCluster.columns),len(genCluster.columns)))
    idx1 = 0
    for genItem in genCluster.columns:
        gSet =  genCluster[genItem].dropna()
        #print(gSet)
        idx2 = 0
        for chemItem in chemCluster.columns:
            cSet = chemCluster[chemItem].dropna()
            #print(cSet)
            result = set(gSet) & set(cSet)
            if(result.__len__() > 1):
                print(result.__len__())
                #print(gSet)
                #print(cSet)
            resultNo[idx1,idx2] = result.__len__()
            idx2+=1
        idx1+=1

    print(resultNo)
    np.savetxt("/home/elliotnam/project/cmap/clusterCompareResult.csv", resultNo, delimiter=",")


def runRankCrunCompareAlgorithmslustering(clustNo):
    rankInfo = pd.read_csv("/home/elliotnam/project/cmap/genedistanceresult_id_column.csv")
    #rankInfo = pd.read_csv("/home/elliotnam/project/cmap/test.csv")
    print(rankInfo.shape)
    jj = rankInfo.values
    print(jj[0:,0:])
    #print(rankInfo.iloc[0,0])
    #kk = (rankInfo.iloc[1:,1:])
    #jj = (rankInfo.iloc[0:,1:]).values
    #print(rankInfo.iloc[0,1])
    #print(rankInfo.iloc[0,2])
    #print(rankInfo.iloc[0,3])
    #print(rankInfo.iloc[1,:])
    #kk = rankInfo.iloc[0:, 1:].astype(float)
    #kk = rankInfo.iloc[0:, 1:]
    #print(kk)
    #kk1 = rankInfo.iloc[1:, 1:]
    #kk2 = rankInfo.iloc[1:, 0:]
    runCluster(clustNo, rankInfo,"/home/elliotnam/project/cmap/genedistanceresult_id_column.pkl", 0)



def runGeneClustering(clustNo):
    rankInfo = pd.read_csv("/home/elliotnam/project/cmap/fingerprintresult.csv")
    print(rankInfo.shape)
    jj = rankInfo.values
    print(jj[0:, 0:])

    runCluster(clustNo, rankInfo, "/home/elliotnam/project/cmap/fingerprintresult.pkl", 0)



if __name__ == "__main__":

    #getSmilesFromPubChem()
    #saveFTFromSmiles()
    #calcFPFromSmiles()
    #calcGeneDistances()
    #seeGenData()

    #runRankClustering(100)
    #runGeneClustering(100)
    #runGenModel("/home/elliotnam/project/cmap/genedistanceresult_id_column")
    #runGenModel("/home/elliotnam/project/cmap/fingerprintresult")
    #showHiraClusterTree("/home/elliotnam/project/cmap/genedistanceresult_id_column.pkl")
    #cleanDrugMiles()
    #testPubChem()
    #compareClusters()
    #checkIsFdaApproved()
    #makeDrugFingerPrintData()
    saveGeneInfo("test")