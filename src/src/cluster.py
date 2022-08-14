import os
import string
import joblib

import numpy as np

from collections import Counter
from sklearn.cluster import k_means_, AgglomerativeClustering as AC
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

true_label_count = [40, 60, 55, 29, 19, 49, 49, 49, 49, 49, 49, 37]
true_label = np.array([], dtype = np.dtype(np.int32))

for i in range(len(true_label_count)):
    true_label = np.append(true_label, np.full(true_label_count[i], i))


#   in all_imgs_Paths: /imgs-compressed/  ->  /imgs/
def compressname_to_originname(compressname, COMPRESS_PATH, ORIGIN_PATH):

    originname = compressname.replace(COMPRESS_PATH, ORIGIN_PATH)
    return originname


def change_name_list(origin_name_list):
    origin_part = '/srtp/'
    new_part = '/srtp2/'
    newlist = []
    for name in origin_name_list:
        newlist.append(name.replace(origin_part, new_part))
    
    return newlist

def purity_core(result, label):
    total_num = len(label)
    cluster_counter = Counter(result)
    original_counter = Counter(label)

    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j:
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)
    return sum(t) / total_num



######################################################################
#                 save the result as txt to target folder 
######################################################################
def save_cluster_result(all_imgs_Paths, index_list, TARGET_FILE_PATH, SUBFOLDER_ORIGIN_PATH, SUBFOLDER_COMPRESS_PATH):
    
    real_class_count = len(index_list)
    #   first remove the original file
    for root,dirs,files in os.walk(TARGET_FILE_PATH):
        for file in files:
            os.remove(TARGET_FILE_PATH+file)

    for i in range(real_class_count):
        FILE_NAME = TARGET_FILE_PATH+'file_'+str(i+1)
        f = open(FILE_NAME,"w")
        #       remove the last '\n'
        
        classi_length = len(index_list[i])
        
        for seq in range(classi_length-1):
            f.write(
                compressname_to_originname( all_imgs_Paths[ index_list[i][seq] ], SUBFOLDER_COMPRESS_PATH, SUBFOLDER_ORIGIN_PATH ) +"\n"
            )
            
        f.write( 
            compressname_to_originname( all_imgs_Paths[ index_list[i][classi_length-1] ], SUBFOLDER_COMPRESS_PATH, SUBFOLDER_ORIGIN_PATH )
        )
        
        
######################################################################
#                 all kinds of cluster method - 1
###################################################################### 
# def cluster_method_1(allWordVectors, MAX_CLASS_COUNT):
#     # return type: [[], []]
    
#     all_features_count = allWordVectors.shape[0]
#     max_score = -2
#     real_class_count = 0
#     best_lables = None

#     def euc_dist(X,Y=None, Y_norm_squared = None, squared = False):
#         return cosine_similarity(X,Y)
#     k_means_.euclidean_distance = euc_dist

#     for i in range(2,MAX_CLASS_COUNT+1):
        
#         final_clf = k_means_.KMeans(n_clusters = i)
#         final_clf.fit(allWordVectors)
#         final_labels = final_clf.labels_

#         this_score = silhouette_score(allWordVectors,final_labels,metric="cosine")

#         print('ncluster = ', str(i).zfill(2), 'score = ', this_score)
#         if this_score > max_score:
#             max_score = this_score
#             best_lables = final_labels.copy()
#             real_class_count = i

#     index_list = []

#     #   construct a series of empty lists
#     for i in range(real_class_count):
#         index_list.append([])

#     #   append each index
#     for i in range(all_features_count):
#         index_list[best_lables[i]].append(i)
    
#     return real_class_count, index_list



######################################################################
#                 all kinds of cluster method - 1
###################################################################### 
def cluster_method_1(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS):
    # return type: [[], []]
    
    all_features_count = allWordVectors.shape[0]
    max_score = -2
    real_class_count = 0
    best_lables = None

    def max(a, b):
        if a > b:   return a
        else:       return b
    

    for i in range(max(3, PREDICT_CLASS_COUNT - TOL_BIAS), PREDICT_CLASS_COUNT + TOL_BIAS + 1):
        final_clf = k_means_.KMeans(n_clusters = i)
        final_clf.fit(allWordVectors)
        final_labels = final_clf.labels_
        
        this_score = calinski_harabasz_score(allWordVectors, final_labels)
        print('ncluster = ', str(i).zfill(2), 'this_score = ', this_score)

        if this_score > max_score:
            max_score = this_score
            best_lables = final_labels.copy()
            real_class_count = i

    index_list = []

    #   construct a series of empty lists
    for i in range(real_class_count):
        index_list.append([])

    #   append each index
    for i in range(all_features_count):
        index_list[best_lables[i]].append(i)

    return real_class_count, index_list

######################################################################
#                 all kinds of cluster method - 2
###################################################################### 
def cluster_method_2(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS):
    # return type: [[], []]
    
    all_features_count = allWordVectors.shape[0]
    max_score = -2
    real_class_count = 0
    best_lables = None

    def euc_dist(X,Y=None, Y_norm_squared = None, squared = False):
        return cosine_similarity(X,Y)
    k_means_.euclidean_distance = euc_dist

    def max(a, b):
        if a > b:   return a
        else:       return b
    

    for i in range(max(3, PREDICT_CLASS_COUNT - TOL_BIAS), PREDICT_CLASS_COUNT + TOL_BIAS + 1):
        final_clf = k_means_.KMeans(n_clusters = i)
        final_clf.fit(allWordVectors)
        final_labels = final_clf.labels_
        
        # intra_score = silhouette_score(allWordVectors,final_labels,metric="cosine")
        # inter_score = purity_core(final_labels, true_label)
        # this_score = intra_score * 1 / 3 + inter_score * 2 /3 
        # print('ncluster = ', str(i).zfill(2), 'this_score = ', this_score, 'intra_score = ', intra_score, 'inter_score = ', inter_score)
        
        this_score = silhouette_score(allWordVectors,final_labels,metric="cosine") 
        print('ncluster = ', str(i).zfill(2), 'this_score = ', this_score)

        if this_score > max_score:
            max_score = this_score
            best_lables = final_labels.copy()
            real_class_count = i

    index_list = []

    #   construct a series of empty lists
    for i in range(real_class_count):
        index_list.append([])

    #   append each index
    for i in range(all_features_count):
        index_list[best_lables[i]].append(i)

    return real_class_count, index_list

######################################################################
#                 all kinds of cluster method - 3
###################################################################### 
def cluster_method_3(allWordVectors, MAX_CLASS_COUNT):
    # return type: [[], []]
    
    all_features_count = allWordVectors.shape[0]
    max_score = -2
    real_class_count = 0
    best_lables = None

    def euc_dist(X,Y=None, Y_norm_squared = None, squared = False):
        return cosine_similarity(X,Y)
    k_means_.euclidean_distance = euc_dist

    for i in range(2,MAX_CLASS_COUNT+1):
        
        final_clf = k_means_.KMeans(n_clusters = i)
        final_clf.fit(allWordVectors)
        final_labels = final_clf.labels_

        this_score_1 = silhouette_score(allWordVectors,final_labels,metric="cosine")
        this_score_2 = calinski_harabasz_score(allWordVectors, final_labels)
        this_score = this_score_1 * this_score_2

        print('ncluster = ', str(i).zfill(2), 'score = ', this_score)
        if this_score > max_score:
            max_score = this_score
            best_lables = final_labels.copy()
            real_class_count = i

    index_list = []

    #   construct a series of empty lists
    for i in range(real_class_count):
        index_list.append([])

    #   append each index
    for i in range(all_features_count):
        index_list[best_lables[i]].append(i)
    
    return real_class_count, index_list

######################################################################
#                 all kinds of cluster method - 4
###################################################################### 
def cluster_method_4(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS):
    # return type: [[], []]
    
    all_features_count = allWordVectors.shape[0]
    min_score = 10
    real_class_count = 0
    best_lables = None

    def euc_dist(X,Y=None, Y_norm_squared = None, squared = False):
        return cosine_similarity(X,Y)
    k_means_.euclidean_distance = euc_dist

    def max(a, b):
        if a > b:   return a
        else:       return b
    

    for i in range(max(3, PREDICT_CLASS_COUNT - TOL_BIAS), PREDICT_CLASS_COUNT + TOL_BIAS + 1):
        
        final_clf = k_means_.KMeans(n_clusters = i)
        final_clf.fit(allWordVectors)
        final_labels = final_clf.labels_
        
        this_score =  davies_bouldin_score(allWordVectors, final_labels) 
        print('ncluster = ', str(i).zfill(2), 'this_score = ', this_score)

        if this_score < min_score:
            min_score = this_score
            best_lables = final_labels.copy()
            real_class_count = i

    index_list = []

    #   construct a series of empty lists
    for i in range(real_class_count):
        index_list.append([])

    #   append each index
    for i in range(all_features_count):
        index_list[best_lables[i]].append(i)

    return real_class_count, index_list

######################################################################
#                 all kinds of cluster method - 5
###################################################################### 
def cluster_method_5(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS, _linkage):
    # return type: [[], []]
    
    all_features_count = allWordVectors.shape[0]
    max_score = -2
    real_class_count = 0
    best_lables = None

    def max(a, b):
        if a > b:   return a
        else:       return b
    

    for i in range(max(3, PREDICT_CLASS_COUNT - TOL_BIAS), PREDICT_CLASS_COUNT + TOL_BIAS + 1):
        final_labels = AC(n_clusters=i,affinity="cosine",linkage=_linkage).fit_predict(allWordVectors)
        
        this_score = calinski_harabasz_score(allWordVectors,final_labels) 
        print('ncluster = ', str(i).zfill(2), 'this_score = ', this_score)

        if this_score > max_score:
            max_score = this_score
            best_lables = final_labels.copy()
            real_class_count = i

    index_list = []

    #   construct a series of empty lists
    for i in range(real_class_count):
        index_list.append([])

    #   append each index
    for i in range(all_features_count):
        index_list[best_lables[i]].append(i)

    return real_class_count, index_list

######################################################################
#                 all kinds of cluster method - 6
###################################################################### 
def cluster_method_6(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS, _linkage):
    # return type: [[], []]
    
    all_features_count = allWordVectors.shape[0]
    max_score = -2
    real_class_count = 0
    best_lables = None

    def max(a, b):
        if a > b:   return a
        else:       return b
    

    for i in range(max(3, PREDICT_CLASS_COUNT - TOL_BIAS), PREDICT_CLASS_COUNT + TOL_BIAS + 1):
        final_labels = AC(n_clusters=i,affinity="cosine",linkage=_linkage).fit_predict(allWordVectors)
        
        this_score = silhouette_score(allWordVectors,final_labels,metric="cosine") 
        print('ncluster = ', str(i).zfill(2), 'this_score = ', this_score)

        if this_score > max_score:
            max_score = this_score
            best_lables = final_labels.copy()
            real_class_count = i

    index_list = []

    #   construct a series of empty lists
    for i in range(real_class_count):
        index_list.append([])

    #   append each index
    for i in range(all_features_count):
        index_list[best_lables[i]].append(i)

    # print(dict(Counter(best_lables)))
    # print(index_list)

    return real_class_count, index_list

######################################################################
#                 all kinds of cluster method - 7
###################################################################### 
def cluster_method_7(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS, _linkage):
    # return type: [[], []]
    
    all_features_count = allWordVectors.shape[0]
    min_score = 10
    real_class_count = 0
    best_lables = None

    def max(a, b):
        if a > b:   return a
        else:       return b
    

    for i in range(max(3, PREDICT_CLASS_COUNT - TOL_BIAS), PREDICT_CLASS_COUNT + TOL_BIAS + 1):
        final_labels = AC(n_clusters=i,affinity="cosine",linkage=_linkage).fit_predict(allWordVectors)
        
        this_score = davies_bouldin_score(allWordVectors,final_labels) 
        print('ncluster = ', str(i).zfill(2), 'this_score = ', this_score)

        if this_score < min_score:
            min_score = this_score
            best_lables = final_labels.copy()
            real_class_count = i

    index_list = []

    #   construct a series of empty lists
    for i in range(real_class_count):
        index_list.append([])

    #   append each index
    for i in range(all_features_count):
        index_list[best_lables[i]].append(i)

    return real_class_count, index_list


######################################################################
#                          entry function
######################################################################
def cluster_imgs(toSaveVecsPath, TARGET_FILE_PATH, cluster_method_seq, score_method_seq, linkage_method, MAX_CLASS_COUNT, PREDICT_CLASS_COUNT, TOL_BIAS):
    allWordVectors, all_imgs_Paths, SUBFOLDER_ORIGIN_PATH, SUBFOLDER_COMPRESS_PATH = joblib.load(toSaveVecsPath)
    # all_imgs_Paths = change_name_list(all_imgs_Paths) only need to use when in folder /srtp2/
    if   cluster_method_seq == 1:
        if      score_method_seq == 1:
            real_class_count, index_list2d = cluster_method_1(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS)
        elif    score_method_seq == 2:
            real_class_count, index_list2d = cluster_method_2(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS)
        elif    score_method_seq == 3:
            real_class_count, index_list2d = cluster_method_4(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS)
    elif cluster_method_seq == 2:
        if      score_method_seq == 1:
            real_class_count, index_list2d = cluster_method_5(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS, linkage_method)
        elif    score_method_seq == 2:
            real_class_count, index_list2d = cluster_method_6(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS, linkage_method)
        elif    score_method_seq == 3:
            real_class_count, index_list2d = cluster_method_7(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS, linkage_method)
    # elif cluster_method_seq == 2:   real_class_count, index_list2d = cluster_method_2(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS)
    # elif cluster_method_seq == 3:   real_class_count, index_list2d = cluster_method_3(allWordVectors, MAX_CLASS_COUNT)
    # elif cluster_method_seq == 4:   real_class_count, index_list2d = cluster_method_4(allWordVectors, MAX_CLASS_COUNT)
    # elif cluster_method_seq == 5:   real_class_count, index_list2d = cluster_method_5(allWordVectors, MAX_CLASS_COUNT, TOL_BIAS, linkage_method)
    # elif cluster_method_seq == 6:   real_class_count, index_list2d = cluster_method_6(allWordVectors, MAX_CLASS_COUNT, TOL_BIAS, linkage_method)
    # elif cluster_method_seq == 7:   real_class_count, index_list2d = cluster_method_7(allWordVectors, MAX_CLASS_COUNT, TOL_BIAS, linkage_method)
    # else:                           real_class_count, index_list2d = cluster_method_2(allWordVectors, PREDICT_CLASS_COUNT, TOL_BIAS)    
    print(all)
    save_cluster_result(all_imgs_Paths, index_list2d, TARGET_FILE_PATH, SUBFOLDER_ORIGIN_PATH, SUBFOLDER_COMPRESS_PATH)
    return real_class_count