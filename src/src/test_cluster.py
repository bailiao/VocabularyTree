import os
import sys
import time
import cluster

time_start = time.time()

MAIN_PATH = os.path.abspath(os.path.dirname(__file__)+'/../')
TO_SAVE_PATH = MAIN_PATH+'/data/vecszip'
TARGET_FILE_PATH = MAIN_PATH+'/out_text/'

DEFAULT_CLUSTER_METHOD = int(sys.argv[1])
DEFAULT_SCORE_METHOD = int(sys.argv[2])
DEFAULT_LINKAGE_METHOD = sys.argv[3]
MAX_CLASS_COUNT = int(sys.argv[4])
PREDICT_CLASS_COUNT = int(sys.argv[5])
TOL_BIAS = int(sys.argv[6])

for root,dirs,files in os.walk(TARGET_FILE_PATH):
    for file in files:
        os.remove(TARGET_FILE_PATH+file)

real_class_count = cluster.cluster_imgs(
    TO_SAVE_PATH,
    TARGET_FILE_PATH,
    DEFAULT_CLUSTER_METHOD,
    DEFAULT_SCORE_METHOD,
    DEFAULT_LINKAGE_METHOD,
    MAX_CLASS_COUNT,
    PREDICT_CLASS_COUNT,
    TOL_BIAS)

time_end = time.time()
time_interval = time_end - time_start
time_interval = round(time_interval,3)
print("Clustering Images Time Cost   : %.3f s"%time_interval)

print("Maximum Number of clusters    :",MAX_CLASS_COUNT)
print("Best Number of clusters       :",real_class_count)



#   command to test cluster

#   make clean
#   python3 ./src/test_cluster.py
#   bash classify