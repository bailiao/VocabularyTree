import cv2
import numpy as np
import features_sample


def extract_from_array(oringin_array, row_list):
    res_list = []
    for i in row_list:
        res_list.append(oringin_array[i].tolist())
    res_array = np.array(res_list)
    return res_array


def extract_oneImg(imgPath, target_features, SAMPLE_MODE, interval_ratio, alpha):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desciptor = sift.detectAndCompute(gray,None)

    seq_list = features_sample.features_Sample(
        desciptor.shape[0], target_features, SAMPLE_MODE, interval_ratio, alpha)
    sample_descriptor = extract_from_array(desciptor,seq_list)
    return sample_descriptor


def extract_mulImgs(imgPaths, target_features, SAMPLE_MODE, interval_ratio, alpha):
    
    FeaturesPerImageCount = []                          # features per img count
    
    #   fix the size of array with the first img
    img = cv2.imread(imgPaths[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, tmp_allFeatures = sift.detectAndCompute(gray,None)              #   allFeatures without sample
    

    sample_seq_list = features_sample.features_Sample(
        tmp_allFeatures.shape[0], target_features, SAMPLE_MODE, interval_ratio, alpha)
    
    real_allFeatures = extract_from_array(tmp_allFeatures,sample_seq_list)   #   allFeatures sampled
    # print(type(real_allFeatures))
    
    FeaturesPerImageCount.append(real_allFeatures.shape[0])

    
    for i in range(1, len(imgPaths) ):
        img = cv2.imread(imgPaths[i] )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = sharpenImg(gray)
        # gray = do_clahe(gray)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desciptor = sift.detectAndCompute(gray,None)

        sample_seq_list = features_sample.features_Sample(
            desciptor.shape[0], target_features, SAMPLE_MODE, interval_ratio, alpha)
        
        sample_descriptor = extract_from_array(desciptor,sample_seq_list)
        real_allFeatures = np.concatenate( (real_allFeatures, sample_descriptor), axis=0 )
        FeaturesPerImageCount.append(sample_descriptor.shape[0])
        
        
    return FeaturesPerImageCount, real_allFeatures