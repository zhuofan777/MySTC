import collections
import os
import pandas as pd
import numpy as np

wenjian = ['0.3_100_3.csv',
           '0.3_100_4.csv',
           '0.3_100_5.csv', '0.3_100_6.csv', '0.3_100_7.csv', '0.3_100_8.csv',
           '0.3_50_3.csv', '0.3_50_4.csv', '0.3_50_5.csv', '0.3_50_6.csv', '0.3_50_7.csv', '0.3_50_8.csv',
           '0.3_60_3.csv', '0.3_60_4.csv', '0.3_60_5.csv', '0.3_60_6.csv', '0.3_60_7.csv', '0.3_60_8.csv',
           '0.3_70_3.csv', '0.3_70_4.csv', '0.3_70_5.csv', '0.3_70_6.csv', '0.3_70_7.csv', '0.3_70_8.csv',
           '0.3_80_3.csv', '0.3_80_4.csv', '0.3_80_5.csv', '0.3_80_6.csv', '0.3_80_7.csv', '0.3_80_8.csv',
           '0.4_100_3.csv', '0.4_100_4.csv', '0.4_100_5.csv', '0.4_100_6.csv', '0.4_100_7.csv', '0.4_100_8.csv',
           '0.4_50_3.csv', '0.4_50_4.csv', '0.4_50_5.csv', '0.4_50_6.csv', '0.4_50_7.csv', '0.4_50_8.csv',
           '0.4_60_3.csv', '0.4_60_4.csv', '0.4_60_5.csv', '0.4_60_6.csv', '0.4_60_7.csv', '0.4_60_8.csv',
           '0.4_70_3.csv', '0.4_70_4.csv', '0.4_70_5.csv', '0.4_70_6.csv', '0.4_70_7.csv', '0.4_70_8.csv',
           '0.4_80_3.csv', '0.4_80_4.csv', '0.4_80_5.csv', '0.4_80_6.csv', '0.4_80_7.csv', '0.4_80_8.csv',
           '0.5_100_3.csv', '0.5_100_4.csv', '0.5_100_5.csv', '0.5_100_6.csv', '0.5_100_7.csv', '0.5_100_8.csv',
           '0.5_50_3.csv', '0.5_50_4.csv', '0.5_50_5.csv', '0.5_50_6.csv', '0.5_50_7.csv', '0.5_50_8.csv',
           '0.5_60_3.csv', '0.5_60_4.csv', '0.5_60_5.csv', '0.5_60_6.csv', '0.5_60_7.csv', '0.5_60_8.csv',
           '0.5_70_3.csv', '0.5_70_4.csv', '0.5_70_5.csv', '0.5_70_6.csv', '0.5_70_7.csv', '0.5_70_8.csv',
           '0.5_80_3.csv', '0.5_80_4.csv', '0.5_80_5.csv', '0.5_80_6.csv', '0.5_80_7.csv', '0.5_80_8.csv',
           '0.6_100_3.csv', '0.6_100_4.csv', '0.6_100_5.csv', '0.6_100_6.csv', '0.6_100_7.csv', '0.6_100_8.csv',
           '0.6_50_3.csv', '0.6_50_4.csv', '0.6_50_5.csv', '0.6_50_6.csv', '0.6_50_7.csv', '0.6_50_8.csv',
           '0.6_60_3.csv', '0.6_60_4.csv', '0.6_60_5.csv', '0.6_60_6.csv', '0.6_60_7.csv', '0.6_60_8.csv',
           '0.6_70_3.csv', '0.6_70_4.csv', '0.6_70_5.csv', '0.6_70_6.csv', '0.6_70_7.csv', '0.6_70_8.csv',
           '0.6_80_3.csv', '0.6_80_4.csv', '0.6_80_5.csv', '0.6_80_6.csv', '0.6_80_7.csv', '0.6_80_8.csv'
           ]
datasets = [
    'ArticularyWordRecognition',
    'AtrialFibrilation',
    'BasicMotions',
    'Cricket',
    'Epilepsy',
    'EthanolConcentration',
    'FingerMovements',
    'HandMovementDirection',
    'Handwriting',
    'Libras',
    'LSST',
    'NATOPS',
    'PenDigits',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'StandWalkJump',
    'UWaveGestureLibrary'
]

ans = []
for file in wenjian:
    file = 'D:/STC/' + file
    accs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with open(file) as f:
        line = f.readline()
        dt = collections.defaultdict(list)
        while line:
            lt = line.split(',')
            dataset = lt[0]
            acc = float(lt[1].strip())
            dt[dataset].append(acc)
            line = f.readline()
        for i in dt.keys():
            avg = sum(dt[i]) / len(dt[i])
            idx = int(datasets.index(i))
            avg = round(avg, 4)
            accs[idx] = avg
        ans.append(accs)
        f.close()
s = 'D://STC//sum.xls'
f = open(s, 'w', encoding='gbk')
line1 = '\t'.join(datasets)
line1 = 'dataset' + '\t' + line1 + '\n'
f.write(line1)
for i in range(len(ans)):
    f.write(wenjian[i])
    f.write('\t')
    for j in range(len(ans[i])):
        f.write(str(ans[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
        f.write('\t')  # 相当于Tab一下，换一个单元格
    f.write('\n')  # 写完一行立马换行

f.close()
# print(ans)
