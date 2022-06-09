import argparse

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import heapq as hq
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import random

parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--alp', type=float, default=0.4)
parser.add_argument('--blt', type=int, default=50)
parser.add_argument('--ylt', type=int, default=5)
args = parser.parse_args()


# 计算方差窗口的大小
def computeWinSize(length, b):
    if 0 < length <= b:
        return 1
    else:
        return length // b


# 计算标准差数组
def computeStdArr(series, b):
    l = len(series)
    # 窗口下取整
    w = computeWinSize(l, b)
    # print(w)
    res = []
    for i in range(w, l - w):
        win_arr = [0 for i in range(2 * w + 1)]
        cnt = 0
        for j in range(i - w, w + i + 1):
            win_arr[cnt] = series[j]
            cnt += 1
        std = np.std(win_arr, ddof=0)
        res.append(std)
    return res


# 选择L*α最大的点的下标,，保留下标
def selectLargePoint(series, a):
    l = len(series)
    # 生成l*a个
    res = hq.nlargest(int(l * a), range(l), key=lambda x: series[x])
    return res


# 判断是否为波峰波谷
def isPeakOrValley(std_arr, index, b):
    win_size = computeWinSize(len(std_arr), b)
    flag = False
    if win_size == 1:
        if ((std_arr[index] > std_arr[index - 1] and std_arr[index] > std_arr[index + 1]) or (
                std_arr[index] < std_arr[index - 1] and std_arr[index] < std_arr[index + 1])):
            flag = True
    elif win_size == 2:
        if ((std_arr[index] >= std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] >= std_arr[
            index + 1] and std_arr[index] > std_arr[index + 2])
                or
                (std_arr[index] <= std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <=
                 std_arr[
                     index + 1]) and std_arr[index] < std_arr[index + 2]):
            flag = True
    elif win_size == 3:
        if ((std_arr[index] >= std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] > std_arr[
            index - 3]
             and std_arr[index] >= std_arr[index + 1] and std_arr[index] > std_arr[index + 2] and std_arr[index] >
             std_arr[
                 index + 3])
                or
                (std_arr[index] <= std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <
                 std_arr[index - 3]) and std_arr[index] <= std_arr[index + 1] and std_arr[index] < std_arr[
                    index + 2] and std_arr[index] < std_arr[index + 3]):
            flag = True
    elif win_size == 4:
        if ((std_arr[index] >= std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] > std_arr[
            index - 3] and std_arr[index] > std_arr[index - 4]
             and std_arr[index] >= std_arr[index + 1] and std_arr[index] > std_arr[index + 2] and std_arr[index] >
             std_arr[
                 index + 3] and std_arr[index] > std_arr[index + 4])
                or
                (std_arr[index] <= std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <
                 std_arr[
                     index - 3] and std_arr[index] < std_arr[index - 4]
                 and std_arr[index] <= std_arr[index + 1] and std_arr[index] < std_arr[index + 2] and std_arr[index] <
                 std_arr[index + 3] and std_arr[index] < std_arr[index + 4])):
            flag = True
    else:
        if ((std_arr[index] > std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] > std_arr[
            index - 3] and std_arr[index] > std_arr[index - 4] and std_arr[index] > std_arr[index - 5]
             and std_arr[index] > std_arr[index + 1] and std_arr[index] > std_arr[index + 2] and std_arr[index] >
             std_arr[
                 index + 3] and std_arr[index] > std_arr[index + 4] and std_arr[index] > std_arr[index + 5])
                or
                (std_arr[index] < std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <
                 std_arr[
                     index - 3] and std_arr[index] < std_arr[index - 4] and std_arr[index] < std_arr[index - 5]
                 and std_arr[index] < std_arr[index + 1] and std_arr[index] < std_arr[index + 2] and std_arr[index] <
                 std_arr[
                     index + 3] and std_arr[index] < std_arr[index + 4] and std_arr[index] < std_arr[index + 5])):
            flag = True
    return flag


# 找到所有合适的区间
def fitPoint(stdA, indexA, b):
    w = computeWinSize(len(stdA), b)
    l = len(indexA)
    res = []
    for i in range(0, l):
        val = indexA[i]
        if w <= val < l - w:
            if isPeakOrValley(stdA, val, b):
                res.append(val + 2 * w + 1)
    return res


# 提取特征关键点
def extractFeaturePoint(arr, a, b):
    stdArrOne = computeStdArr(arr, b)
    stdArrTwo = computeStdArr(stdArrOne, b)
    largePoint = selectLargePoint(stdArrTwo, a)
    indexArr = fitPoint(stdArrTwo, largePoint, b)
    return indexArr


# 判断有几个关键点与它相近
def closeNums(keyList, key, threshold):
    cnt = 0
    cnt += keyList.count(key)
    for i in range(1, threshold):
        a = key + i
        b = key - i
        cnt += keyList.count(a)
        cnt += keyList.count(b)
    return cnt


class keyPoint:
    def __init__(self, key, cnt, dim):
        self.key = key
        self.cnt = cnt
        self.dim = dim


# 找到较为相近的关键点
def selectKeys(data, alp, blt, ylt):
    # 原始data顺序 样本数 维度数 序列长度
    dim_nums = data.shape[1]
    sample_nums = data.shape[0]
    all_keys = []
    for sm in tqdm(range(sample_nums)):
        one_sample = data[sm]
        key_points = []
        key_list = []
        ex_key_points = []
        for dn in range(dim_nums):
            keys = extractFeaturePoint(one_sample[dn], alp, blt)
            key_points.append(keys)
            key_list.extend(keys)
        for dn in range(dim_nums):
            kl = key_points[dn]
            ekl = []
            for k in kl:
                cnt = closeNums(key_list, k, ylt)
                ekl.append(keyPoint(k, cnt, dn))
            ex_key_points.extend(ekl)
        # 取前30%的70% 和 后70%的30%
        ex_key_points.sort(key=lambda x: x.cnt, reverse=True)
        # print(len(ex_key_points))
        length = len(ex_key_points)
        # 前30%
        preLength = int(0.3 * length)
        preKey = ex_key_points[:preLength + 1]
        # # 后70%
        # aftLength = length - preLength
        # aftKey = ex_key_points[preLength+1:]
        # 取前30%的70%
        # preKey = random.sample(preKey,int(preLength * 0.7))
        # aftKey = random.sample(aftKey,int(aftLength * 0.3))
        ex_key_points = []
        ex_key_points.extend(preKey)
        # ex_key_points.extend(aftKey)
        # print(len(ex_key_points))
        all_keys.append(ex_key_points)
    return all_keys


class shapelet:
    def __init__(self, keyPoint, ylt, st, ed, length, val, sample):
        # keyPoint 由那个keyPoint生成
        # ylt 松弛是多少
        # st 起始位置
        # ed 终止位置
        # val shapelet在序列中的实际的值
        self.quality = None
        self.keyPoint = keyPoint
        self.ylt = ylt
        self.st = st
        self.ed = ed
        self.length = length
        self.val = val
        self.sample = sample

    def setQuality(self, quality):
        self.quality = quality


def generateCandidates(s, sm, l, keyP, ylt):
    # sm 为单个样本[维度，长度]
    # keyP 应该为一个keyPoint
    # ylt为松弛参数
    candidate = []
    if l % 2 == 1:
        l = l // 2
        dim = keyP.dim
        s = s[dim]
        for i in range(-ylt, ylt + 1):
            # 从中心点两侧的松弛长度进行计算
            if keyP.key - i - l < 0:
                continue
            left = keyP.key - i - l
            if keyP.key - i + l + 1 >= len(s):
                continue
            right = keyP.key - i + l + 1
            spt = shapelet(keyPoint=keyP, ylt=ylt, st=left, ed=right - 1, length=right - left, val=s[left:right],
                           sample=sm)
            candidate.append(spt)
    else:
        l = l // 2
        dim = keyP.dim
        s = s[dim]
        for i in range(-ylt, ylt + 1):
            # 从中心点两侧的松弛长度进行计算
            if keyP.key - i - l < 0:
                continue
            left = keyP.key - i - l
            if keyP.key - i + l >= len(s):
                continue
            right = keyP.key - i + l
            spt = shapelet(keyPoint=keyP, ylt=ylt, st=left, ed=right - 1, length=right - left, val=s[left:right],
                           sample=sm)
            candidate.append(spt)
            if keyP.key - i - l + 1 < 0:
                continue
            left = keyP.key - i - l + 1
            if keyP.key - i + l + 1 >= len(s):
                continue
            right = keyP.key - i + l + 1
            spt = shapelet(keyPoint=keyP, ylt=ylt, st=left, ed=right - 1, length=right - left, val=s[left:right],
                           sample=sm)
            candidate.append(spt)
    return candidate


def findShapeletCandidates(data, lmin, lmax, keys, ylt):
    sample_nums = data.shape[0]
    candidate = []
    for sm in tqdm(range(sample_nums)):
        s = data[sm]
        keyList = keys[sm]
        for k in keyList:
            # lmin - lmax
            for l in range(lmin, lmax + 1):
                candidate.extend(generateCandidates(s, sm, l, k, ylt))
    # print(len(candidate))
    dt = collections.defaultdict(list)
    for cd in candidate:
        dt[cd.length].append(cd)
    return dt
    # print(dt.keys())


def findDistances(c, l, data):
    # 查找获选shapelet在每个样本上的最小值
    ds = []
    dim = c.keyPoint.dim
    data = np.transpose(data, axes=(1, 0, 2))
    sers = data[dim]
    for s in sers:
        cp = [s[i: i + l] for i in range(0, len(s) - l + 1, 1)]
        ds.append(cdist([c.val], cp, metric='seuclidean').min())
    return np.array(ds)


def assessCandidate(ds, label):
    class_groups = []
    for c in np.unique(label):
        class_groups.append(ds[label == c].tolist())
    #  返回f值
    return f_oneway(*class_groups).statistic


def sortByQuality(shapelets):
    return sorted(shapelets, key=lambda s: s.quality, reverse=True)


def removeSelfSimilar(shapelets):
    ts = shapelets[:][::-1]
    it = []
    for x in ts:
        it.append(pd.Interval(x.st, x.ed, closed='both'))
    removeIdx = []
    l = len(it)
    for i in range(l):
        for j in range(i + 1, l):
            if it[i].overlaps(it[j]):
                removeIdx.append(i)
                break
    res = []
    for i in range(l):
        if i not in removeIdx:
            res.append(ts[i])
    return res[::-1]


def merge(k, kShapelets, shapelets):
    total_shapelets = kShapelets + shapelets
    return sortByQuality(total_shapelets)[:k]


def findShapelets(data, label, dt, k):
    kShapelets = []
    # l 为长度
    for l in dt.keys():
        # cd 是单个长度的所有样本和维度下的shaplet候选
        cd = dt[l]
        for i, c in enumerate(tqdm(cd)):
            ds = findDistances(c, l, data)
            quality = assessCandidate(ds, label)
            cd[i].setQuality(quality)
        cd = sortByQuality(cd)
        # print(len(cd))
        cd = removeSelfSimilar(cd)
        kShapelets = merge(k, kShapelets, cd)
    return kShapelets
    # print(len(cd))
    # print(cd[0].quality,cd[1].quality,cd[-1].quality)
    # print(k)


def load_raw_ts(path, dataset):
    path = path + "raw//" + dataset + "//"
    # 训练集
    x_train = np.load(path + 'X_train.npy')
    x_train = np.transpose(x_train, axes=(0, 2, 1))
    x_test = np.load(path + 'X_test.npy')
    x_test = np.transpose(x_test, axes=(0, 2, 1))
    y_train = np.load(path + 'y_train.npy')
    y_test = np.load(path + 'y_test.npy')
    labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(labels)) + 1
    return x_train, x_test, y_train.reshape(-1), y_test.reshape(-1), nclass


def run(dgo, alp, blt, ylt):
    # target_train 为标签 data_train 为序列
    data_train, data_test, target_train, target_test, nclass = load_raw_ts("D://tmppro//data//", dgo)
    sample_nums = data_train.shape[0]
    dim_nums = data_train.shape[1]
    series_length = data_train.shape[2]
    test_nums = data_test.shape[0]
    lmin = 5
    lmax = 13
    # 选取shapelet的数量
    k = int(series_length / 2)
    # alp = 0.4
    # # k = 20
    # # alp = random.choice([0.3, 0.4])
    # # blt = random.choice([100, 90, 80, 70, 60, 50])
    # blt = 50
    # # ylt = random.choice([5,6,7,8])
    # ylt = 5
    keys = selectKeys(data_train, alp, blt, ylt)
    dt = findShapeletCandidates(data_train, lmin, lmax, keys, ylt)
    kshapelets = findShapelets(data_train, target_train, dt, k)
    dataset = np.zeros((sample_nums, k))
    for i, sp in enumerate(tqdm(kshapelets)):
        Ds = findDistances(sp, sp.length, data_train)
        dataset[:, i] = Ds
    # print(dataset)
    dataset_test = np.zeros((test_nums, k))
    for i, sp in enumerate(tqdm(kshapelets)):
        Ds = findDistances(sp, sp.length, data_test)
        dataset_test[:, i] = Ds
    sc = StandardScaler()
    # model = LogisticRegressionCV().fit(sc.fit_transform(dataset),target_train)
    model = LinearSVC().fit(sc.fit_transform(dataset), target_train)
    print(classification_report(target_train, model.predict(sc.transform(dataset))))
    print(accuracy_score(target_train, model.predict(sc.transform(dataset))))
    print(classification_report(target_test, model.predict(sc.transform(dataset_test))))
    sc = accuracy_score(target_test, model.predict(sc.transform(dataset_test)))
    print(sc)
    s = 'D://STC//'+str(alp)+'_'+str(blt)+'_'+str(ylt)+'.csv'
    f = open(s, 'a')
    f.write(dgo + ',' + str(sc) + ',' + str(alp) + ',' + str(blt) + ',' + str(ylt) + ',' + '\n')


dgos = [ 'Libras', 'AtrialFibrilation', 'BasicMotions', 'RacketSports', 'Handwriting', 'Epilepsy',
        'JapaneseVowels', 'UWaveGestureLibrary', 'PenDigits',
        'StandWalkJump', 'NATOPS', 'ArticularyWordRecognition', 'FingerMovements'
        # ,'LSST'
        # ,'HandMovementDirection'
        ]

for i in dgos:
    try:
        run(i, args.alp, args.blt, args.ylt)
    except:
        print(i, " has an error")
