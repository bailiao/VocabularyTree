import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import declaration
import features_extract

DEFAULT_MINIMUM_CLUSTER_SIZE = declaration.DEFAULT_MINIMUM_CLUSTER_SIZE
DEFAULT_DEPTH = declaration.DEFAULT_DEPTH
DEFAULT_BRANCH = declaration.DEFAULT_BRANCH

class vocabularyTreeNode():
    def __init__(self,branchNum, nFeatures=0,features=None):
        self.real_nBranch = min(branchNum, 
                                nFeatures//DEFAULT_MINIMUM_CLUSTER_SIZE)        
                                            # 每个节点的实际分支数目,依据该结点的特征数量决定
        self.nFeatures = nFeatures          # 分支下面所有节点的总数
        self.feature = features             # 该节点所有向量的平均值
        self.weight = 0.0                   # 节点在最终词向量中的权重
        self.children = []                  # 子节点
        #Term Frequency
        self.tf = 0.0

#class featuresArraying():

# def Distance(list1,list2):
#     distance = 0.0
#     for i in range(len(list1)):
#         distance += (list1[i] - list2[i])*(list1[i] - list2[i])
#     return distance


# leafCount = 0
class vocabularyTree():
    def __init__(self, _depth=DEFAULT_DEPTH, _nBranch=DEFAULT_BRANCH, _root=None):
        self.root = _root                   # 树的根节点
        # self.nNodes = 0                     # 树的所有节点?
        self.nBranch = _nBranch             # 树的最大分支个数
        self.depth = _depth                 # 树的最大高度
    
    def buildTree(self, featuresArray, nFeatures, nBranch):
        rootFeature = []
        self.root = vocabularyTreeNode(nBranch,nFeatures,rootFeature)
        self.buildRecursion(self.root, 0, featuresArray, nFeatures)
        return

    # in this function, self always represent the whole tree itself
    # self.depth can always be used to represent the maximum depth
    # self.branch can alsays be used to represent the maximum number of branches
    def buildRecursion(self, curNode, curDepth, featuresArray, nFeatures):
        if curDepth  == self.depth:     return
        if nFeatures == 0:              return
        if curNode.real_nBranch <= 1:   return      # 如果少于两枝,不进行继续划分

        # //    After an ndarray is clustered by KMeans, it remains the same
        # //    一个数组聚类之后保持不变\
        
        clf = KMeans(n_clusters = curNode.real_nBranch)
        clf.fit(featuresArray)                      # 分组
                
        # 这里还不知道每一个cluster中分别有多少个向量,但是可以给每个节点中的平均值赋值
        # for tmpCenter in clf.cluster_centers_:
        #     curNode.children.append(vocabularyTreeNode(0, self.nBranch, tmpCenter))

        labels = clf.labels_            # 每个数据点所属分组

        newFeaturesArray = []
        for i in range(0,curNode.real_nBranch):
            newFeaturesArray.append([])
        
        CountArray = np.zeros(curNode.real_nBranch,dtype=int)

        for i in range(0,nFeatures):
            CountArray[labels[i]] += 1
            newFeaturesArray[labels[i]].append(featuresArray[i].tolist())
        
        for i in range(0,curNode.real_nBranch):
            curNode.children.append(vocabularyTreeNode(self.nBranch,CountArray[i],clf.cluster_centers_[i]))
            self.buildRecursion(curNode.children[i], curDepth+1, np.array(newFeaturesArray[i]), CountArray[i])

    def DFS(self, curDepth, curNode):
        # if curDepth == self.depth:      return
        print(curNode.real_nBranch)
        if self.isLeaf(curNode) == True:        
            print("------------------------------------------\n")
            return
        for i in range(curNode.real_nBranch):
            self.DFS(curDepth+1,curNode.children[i])
    
    def getTFIDFVector(self, features, nImages, Features_PerImage_Array):
        tfidfVector = []
        featureCount = 0
        for index in range(nImages):        # 得到所有图片的初始向量表示
            oneImgTFIDF = self.getOneTFIDFVector(features, Features_PerImage_Array[index], featureCount)
            tfidfVector.append(oneImgTFIDF)
            featureCount += Features_PerImage_Array[index]
        
        #叶节点数量，即描述向量的长度
        valid_feature_count = len(tfidfVector[0])
        weightList = [0 for i in range(valid_feature_count)]
        # sumtmp=0
        for oneImgTFIDF in tfidfVector :
            for tf, index in zip(oneImgTFIDF, range(valid_feature_count)):
                # sumtmp += tf
                weightList[index] += tf > 0
        # print("sum=",end="")
        # print(sumtmp)

        for index in range(valid_feature_count):
            if weightList[index] != 0.0:
                weightList[index] = np.log(nImages) - np.log(weightList[index])
        
        for i in range(nImages) :
            for j in range(valid_feature_count):
                tfidfVector[i][j] *= weightList[j]

        return tfidfVector.copy(), weightList.copy()


    def getOneTFIDFVector(self, features, ImageFeatureNum, start):  # 对于一个图片来说，把所有的特征向量走一遍HKAdd流程
        self.ClearTF(self.root)                 # 0初始化
        for index in range(ImageFeatureNum):    # 所有图片在树中遍历
            self.HKAdd(features[start+index], self.root)
        oneImgTFIDF = []
        self.getTFIDF(oneImgTFIDF, self.root)
        return oneImgTFIDF.copy()

    def HKAdd(self, feature, curNode):       # 拿到一个向量，从根节点到叶节点的路径+1   # feature类型: ndarray
        curNode.tf += 1
        if self.isLeaf(curNode) == True:    return
        else:
            minIndex = -1
            minDis = 10000000
            # if curNode.nFeatures > curNode.real_nBranch:
            for child, index in zip(curNode.children, range(len(curNode.children) ) ):
                curDis = np.linalg.norm(feature - np.array(child.feature))
                if curDis < minDis:
                    minDis = curDis
                    minIndex = index
            self.HKAdd(feature, curNode.children[minIndex])

    # def getTFIDF(self, tfList, curNode):    # 前序遍历一棵树，从左到右得到所有的叶子节点的tf值
    #     if self.isLeaf(curNode): 
    #         tfList.append(curNode.tf)
    #     else:
    #         for child in curNode.children:
    #             self.getTFIDF(tfList, child)

    def getTFIDF(self, tfList, curNode):    # 前序遍历一棵树，得到所有节点的tf值
        tfList.append(curNode.tf)
        if self.isLeaf(curNode) == False: 
            for child in curNode.children:
                self.getTFIDF(tfList, child)

    def ClearTF(self, curNode):
        curNode.tf = 0.0
        if self.isLeaf(curNode):    return
        else:
            for child in curNode.children:
                self.ClearTF(child)    

    def isLeaf(self, curNode):
        leaf = False
        if curNode.children == []:
            leaf = True
        return leaf
    

    def searchImageVector(self, imgPath, target_features, SAMPLE_MODE, interval_ratio, alpha):
        oneImgTFIDF = []
        if os.path.exists(imgPath) == False:     
            return oneImgTFIDF
        
        descriptor = features_extract.extract_oneImg(
            imgPath, target_features, SAMPLE_MODE, interval_ratio, alpha)
        
        self.ClearTF(self.root)                         #   0初始化
        for index in range(descriptor.shape[0]):        #   所有图片在树中遍历
            self.HKAdd(descriptor[index], self.root)
        
        self.getTFIDF(oneImgTFIDF, self.root)
        return oneImgTFIDF.copy()