import os

#########################################
#      for extract & sample features
#########################################
DEFAULT_FEATURE_EXTRACT = 1000      #   decide the ratio of sample for each img
DEFAULT_BUILD_TREE_RATIO = 0.4      #   构建词汇树随机选取的图片占据总图片数比例
DEFAULT_SAMPLE_MODE = 1     
                                    #   1   from small to big (DEFAULT)
                                    #   2   from big to small
                                    #   3   right-side-normal distribution
     
DEFAULT_INTERVAL_RATIO = 2**0.5     #   the ratio of length of two intervals
DEFAULT_ALPHA = 0.09                #   DEFAULT len ration of the first interval of sample

#########################################
#            for build tree
#########################################
DEFAULT_BRANCH = 150
DEFAULT_DEPTH = 10
DEFAULT_LENGTH = 128    
DEFAULT_MINIMUM_CLUSTER_SIZE = 40
DEFAULT_PARTIAL_BUILD_TREE = 100

#########################################
#           absolute  paths
#########################################

MAIN_PATH = os.path.abspath(os.path.dirname(__file__)+'/../')
SUBFOLDER_IMG_PATH = '/imgs-compressed/' 
SUBFOLDER_SAVE_DATA_PATH = '/data/'
SUBFOLDER_TARGET_FILE_PATH = '/out_text/'

IMG_PATH = MAIN_PATH + SUBFOLDER_IMG_PATH                   # build tree from this folder
SAVE_DATA_PATH = MAIN_PATH + SUBFOLDER_SAVE_DATA_PATH
TARGET_FILE_PATH = MAIN_PATH + SUBFOLDER_TARGET_FILE_PATH

#########################################
#           for compress imgs
#########################################

DEFAULT_SIZE_CUT = 8                #   pic size threshold
DEFAULT_COMPRESS_QUALITY = 95       #   compress quality
DEFAULT_RESIZED_RATIO = 2           #   resized pic ratio

SUBFOLDER_ORIGIN_PATH   = '/imgs/'
SUBFOLDER_COMPRESS_PATH = '/imgs-compressed/'
DEFAULT_ORIGIN_PATH   = MAIN_PATH + SUBFOLDER_ORIGIN_PATH
DEFAULT_COMPRESS_PATH = MAIN_PATH + SUBFOLDER_COMPRESS_PATH

#########################################
#           for clustering imgs
#########################################
DEFAULT_CLUSTER_METHOD = 2