from sympy.solvers import nsolve
from sympy import Symbol
import numpy as np
import math
import random
import scipy.stats as stats


#   return list -- 每个区间和第一个区间长度之比
def getSplitList(alpha, q, estimated_value):
    x = Symbol('x')
    n = int( round( nsolve(alpha*(1-q**x) - 1 + q, x, estimated_value) ) )  # 需要给出估计值才能正确计算
    spList = np.logspace(0,n-1,n,base=q).tolist()
    return spList

#   0 - len
#   前面区间长度小, 后面区间长度大
def smallTobig_split(descriptor_seqlist, spList, alpha, nFeatures):
    result = []
    nIntervals = len(spList)
    descriptor_seqlist_len = len(descriptor_seqlist)
    begin = 0
    for index in range(0, nIntervals):      #   
        interval_len = math.floor( descriptor_seqlist_len*alpha*spList[index] )
        end = begin + interval_len
        
        if end > descriptor_seqlist_len:   end = descriptor_seqlist_len
        real_sample_count = min(nFeatures//nIntervals, end - begin)                 #   实际的采样长度小于区间长度
        
        result += random.sample(descriptor_seqlist[begin:end], real_sample_count)   #  区间内随机采样
        
        begin = end
        if begin == descriptor_seqlist_len:
            break
    
    return result

#   0 - len
#   前面区间长度大, 后面区间长度小
def bigTosmall_split(descriptor_seqlist, spList, alpha, nFeatures):
    result = []
    nIntervals = len(spList)
    descriptor_seqlist_len = len(descriptor_seqlist)
    begin = 0
    for index in range(len(spList)-1, -1, -1):  
        interval_len = math.floor( descriptor_seqlist_len*alpha*spList[index] )
        end = begin + interval_len
        
        if end > descriptor_seqlist_len:   end = descriptor_seqlist_len
        real_sample_count = min(nFeatures//nIntervals, end - begin)                 #   实际的采样长度小于区间长度
        
        result += random.sample(descriptor_seqlist[begin:end], real_sample_count)   #  区间内随机采样
        
        begin = end
        if begin == descriptor_seqlist_len:
            break
    
    return result

#   正态分布，取右半部分
def normal_split(descriptor_seqlist, nFeatures):
    descriptor_seqlist_len = len(descriptor_seqlist)
    mu = 0
    sigma = descriptor_seqlist_len / 3
    X = stats.truncnorm( 0, (descriptor_seqlist_len-1 - mu) / sigma, loc=mu, scale=sigma ) #有区间限制的随机数

    s = X.rvs(nFeatures)
    s = [int(round(i)) for i in s]
    result = [descriptor_seqlist[i] for i in s]
    return result


def features_Sample(total_features, target_features, SAMPLE_MODE, interval_ratio, alpha):
    default_guess = 5
    interval_list = getSplitList(alpha, interval_ratio, default_guess)
    seqlist = [i for i in range(total_features)]
    if   SAMPLE_MODE == 1:  return smallTobig_split(seqlist,interval_list,alpha,target_features)
    elif SAMPLE_MODE == 2:  return bigTosmall_split(seqlist,interval_list,alpha,target_features)
    else:                   return normal_split(seqlist,target_features)
