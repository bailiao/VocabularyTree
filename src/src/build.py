import os
import sys
import time
import random
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import declaration
import img_compress
import features_extract
import vocabulary_tree
import cluster


MAX_CLASS_COUNT = int(sys.argv[1])
PREDICT_CLASS_COUNT = int(sys.argv[2])
TOL_BIAS = int(sys.argv[3])


#   first check whether classify has been done
classify_done = os.path.exists(declaration.IMG_PATH+'/class_1')
if not classify_done:   # do not classify
    
# ------------------------------------------------------------------------ #
#                         compress all pictures 
# ------------------------------------------------------------------------ #
    time_start = time.time()
    img_compress.compressImgs(
        declaration.DEFAULT_SIZE_CUT,
        declaration.DEFAULT_COMPRESS_QUALITY,
        declaration.DEFAULT_RESIZED_RATIO, 
        declaration.DEFAULT_ORIGIN_PATH, 
        declaration.DEFAULT_COMPRESS_PATH)
    
    time_end = time.time()
    time_interval = time_end - time_start
    time_interval = round(time_interval,3)
    print("-----------------------------------------------------------------------------------")
    print("Compressing Images Time Cost  : %.3f s"%time_interval)
    
# ------------------------------------------------------------------------ #
#              readin all pictures & extract their features 
# ------------------------------------------------------------------------ #
    time_start = time.time()
    
    #  -------------------- readin imgs -------------------------- #
    all_imgs_Paths = []   
    for root,dirs,files in os.walk(declaration.IMG_PATH):
        for file in files: 
            all_imgs_Paths.append(os.path.join(root,file)) 
    
    all_imgs_seq_list = [i for i in range(len(all_imgs_Paths))] #   建树用到的imgpath对应的序号

    if len(all_imgs_Paths) > declaration.DEFAULT_PARTIAL_BUILD_TREE:
        build_img_seq_list = random.sample(
            all_imgs_seq_list, 
            int(declaration.DEFAULT_BUILD_TREE_RATIO*len(all_imgs_seq_list)) )   
    else:
        build_img_seq_list = all_imgs_seq_list
    #  ------------------- extract features  --------------------- #

    build_imgs_Paths = []
    for i in build_img_seq_list:
        build_imgs_Paths.append(all_imgs_Paths[i])
    
    FeaturesPerImageCount, real_allFeatures = features_extract.extract_mulImgs(
        build_imgs_Paths,
        declaration.DEFAULT_FEATURE_EXTRACT,
        declaration.DEFAULT_SAMPLE_MODE,
        declaration.DEFAULT_INTERVAL_RATIO,
        declaration.DEFAULT_ALPHA)
        # features per img count
                    
    
    #  ------------------- standardize the vector  --------------------- #

    all_features_data = pd.DataFrame(real_allFeatures)
    scaler = StandardScaler()
    all_features_data = scaler.fit_transform(all_features_data)
    real_allFeatures = np.array(all_features_data)      # real_allFeatures: type np-array

    time_end = time.time()
    time_interval = time_end - time_start
    time_interval = round(time_interval,3)
    
    print("Total-imgs count              :", len(all_imgs_Paths))
    print("Build-tree-imgs count         :", len(build_img_seq_list))
    print("Max features per image        :", declaration.DEFAULT_FEATURE_EXTRACT)
    print("Total sampled-features count  :", real_allFeatures.shape[0])

    print("Features Extracting Time Cost : %.3f s"%time_interval)
    
# ------------------------------------------------------------------------ #
#                         build & save the tree
# ------------------------------------------------------------------------ #
    time_start = time.time()
    FinalTree = vocabulary_tree.vocabularyTree(
        declaration.DEFAULT_DEPTH,
        declaration.DEFAULT_BRANCH)

    totalFeaturesCount = real_allFeatures.shape[0]
    FinalTree.buildTree(
        real_allFeatures,
        totalFeaturesCount,
        declaration.DEFAULT_BRANCH) 

    toSaveTree = joblib.dump(FinalTree, declaration.SAVE_DATA_PATH+'vtree')
    
    time_end = time.time()
    time_interval = time_end - time_start
    time_interval = round(time_interval,3)
    print("Building Database Time Cost   : %.3f s"%time_interval)
    
# ------------------------------------------------------------------------ #
#                             get weight list
# ------------------------------------------------------------------------ #
    time_start = time.time()
    
    tfidfVector,weightList = FinalTree.getTFIDFVector(
        real_allFeatures, 
        len(build_img_seq_list), 
        FeaturesPerImageCount)

    time_end = time.time()
    time_interval = time_end - time_start
    time_interval = round(time_interval,3)
    print("Getting Weight List Time Cost : %.3f s"%time_interval)

# ------------------------------------------------------------------------ #
#                      get the word_vector for all the Images
# ------------------------------------------------------------------------ #
    time_start = time.time()
    allWordVectors = []
    
    for i in range(len(all_imgs_Paths)):      #   get all vectors
        
        searchVector = FinalTree.searchImageVector(
            all_imgs_Paths[i],
            declaration.DEFAULT_FEATURE_EXTRACT,
            declaration.DEFAULT_SAMPLE_MODE,
            declaration.DEFAULT_RESIZED_RATIO,
            declaration.DEFAULT_ALPHA)             # search img i
        
        for weighti in range(len(weightList)):                              # adjust with weight
            searchVector[weighti] *= weightList[weighti]
        allWordVectors.append(searchVector)                                 # append this vector to list
    
    # ---------------- standardize word vecs------------------ #
    allWordVectors = np.array(allWordVectors)
    vec_data = pd.DataFrame(allWordVectors)
    scaler = StandardScaler()
    vec_data = scaler.fit_transform(vec_data)
    allWordVectors = np.array(vec_data)     # allWordVectors-type: np.array
    
    toSaveVecsPath = declaration.SAVE_DATA_PATH+'vecszip'
    toSaveVecs = joblib.dump(
        (allWordVectors, all_imgs_Paths, declaration.SUBFOLDER_ORIGIN_PATH, declaration.SUBFOLDER_COMPRESS_PATH),
        toSaveVecsPath)

    time_end = time.time()
    time_interval = time_end - time_start
    time_interval = round(time_interval,3)
    print("Getting Vectors for all imgs  : %.3f s"%time_interval)
    
# ------------------------------------------------------------------------ #
#                      get the best cluster result
# ------------------------------------------------------------------------ #
    time_start = time.time()

    
    real_class_count = cluster.cluster_imgs(
        toSaveVecsPath,
        declaration.TARGET_FILE_PATH,
        declaration.DEFAULT_CLUSTER_METHOD,
        MAX_CLASS_COUNT,
        PREDICT_CLASS_COUNT,
        TOL_BIAS)

    time_end = time.time()
    time_interval = time_end - time_start
    time_interval = round(time_interval,3)
    print("Clustering Images Time Cost   : %.3f s"%time_interval)
    
    print("Maximum Number of clusters    :",MAX_CLASS_COUNT)
    print("Best Number of clusters       :",real_class_count)

