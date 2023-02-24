import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

def plot_result(Ver, Met, Time, ext, Name=''):
    N = len(ext)
    Met, Time = np.array(Met), np.array(Time)
    Ver, Met, Time = [Ver[i] for i in ext], Met[:,ext], Time[:,ext]
    idx = np.arange(N) * 4
    bar_width = 0.8
    Bot = [[0 for _ in range(N)] for _ in range(5)]
    for i in range(4):
        for j in range(N):
            if i == 0:
                Bot[i + 1][j] = Time[i][j]
            else:
                Bot[i + 1][j] = Bot[i][j] + Time[i][j]
    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.grid(True)
    fig.set_facecolor('white')
    b1 = plt.bar(idx, Met[0], bar_width, label='HMean', color='black', zorder=10)
    b2 = plt.bar(idx + bar_width, Met[1], bar_width, label='Precision', color='red', zorder=10)
    b3 = plt.bar(idx + 2*bar_width, Met[2], bar_width, label='Recall', color='blue', zorder=10)
    plt.xticks(idx + 3/2*bar_width, Ver, rotation=0)
    plt.ylabel('[%]')
    plt.ylim([96.5, 98])
    ax2 = ax1.twinx()
    b4 = ax2.bar(idx + 3*bar_width, Time[0], label='Data handling\n(Mask, GPU to CPU)', color='silver')
    b5 = ax2.bar(idx + 3*bar_width, Time[1], bottom=Bot[1], label='PA (Pixel Aggregation)', color='indigo')
    b6 = ax2.bar(idx + 3*bar_width, Time[2], bottom=Bot[2], label='Resizing', color='magenta')
    b7 = ax2.bar(idx + 3*bar_width, Time[3], bottom=Bot[3], label='Boxgen', color='olive')
    b8 = ax2.bar(idx + 3*bar_width, Time[4], bottom=Bot[4], label='Data handling (Output)', color='cyan')
    plt.ylim([0, 1500])
    plt.ylabel('Time [ms]')
    bs = [b1, b2, b3, b4, b5, b6, b7, b8]
    labels = [b.get_label() for b in bs]
    plt.legend(bs, labels, loc='upper right', bbox_to_anchor=(1.41, 1.02))
    if Name != '':
        plt.savefig(Name + '.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=False)
        
def compareIMG(img, cut, Name='', *Ver):
    if len(Ver) > 4:
        return
    elif len(Ver) == 2:
        fig = plt.figure(figsize=(15, 15))
        fig.set_facecolor('white')
        for i, j in enumerate(Ver):
            plt.subplot(1,2,i+1)
            tmp = plt.imread('../../../../outputs/' + j + '/' + img)
            plt.imshow(tmp[cut[0]:cut[0]+cut[2],cut[1]:cut[1]+cut[3],:])
            plt.title(j)
            plt.axis('off')
        if Name != '':
            plt.savefig(Name + '.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=False)
    elif len(Ver) == 4:
        fig = plt.figure(figsize=(15, 15))
        fig.set_facecolor('white')
        for i, j in enumerate(Ver):
            plt.subplot(2,2,i+1)
            tmp = plt.imread('../../../../outputs/' + j + '/' + img)
            plt.imshow(tmp[cut[0]:cut[0]+cut[2],cut[1]:cut[1]+cut[3],:])
            plt.title(j)
            plt.axis('off')
        if Name != '':
            plt.savefig(Name + '.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set Mode')
    parser.add_argument('--mode')
    parser.add_argument('--compare')
    parser.add_argument('--name')
    args = parser.parse_args()
    print(args)
    if args.mode == '0':
        data = pd.read_csv('../time/time.csv', header=None)
        K = data[0][:]
        N = len(K)
        Ver = ['' for _ in range(N)]
        Time = [[0 for _ in range(N)] for _ in range(5)]
        met = {}
        Met = [[0 for _ in range(N)] for _ in range(3)]
        for i in range(N):
            Ver[i] = K[i]
            for j in range(5):
                Time[j][i] = data[j+1][i]
        for i in range(N):
            data = pd.read_csv('../evaluation/' + Ver[i] + '.csv', header=None)        
            for j in range(3):
                Met[j][i] = data[j+1][0]
        if args.compare == None:
            plot_result(Ver, Met, Time, list(range(N)), 'Result')
        else:
            tmp = []
            for i in map(str, args.compare[:-1].split(',')):
                tmp.append(Ver.index(i))
            plot_result(Ver, Met, Time, tmp, args.name)
    elif args.mode == '1':
        os.chdir('../../outputs/Ground_Truth')
        data = []
        for i in os.listdir():
            if not '.txt' in i:
                data.append(i)
        os.chdir('../../results/Results/compare')
        try:
            os.mkdir(args.compare)
        except:
            pass
        os.chdir(args.compare)
        a = 'Ground_Truth'
        b = 'Core'
#         c = 'TwinReader'
#         d = '2_0.2_0.5'
        boxSize = 500
        for i in range(10):
            for j in range(5):
                compareIMG(data[int(args.compare)], [i*100,j*100,boxSize,boxSize],
                           data[int(args.compare)][:-4]+'-'+str(i*100)+'-'+str(j*100),
                           a, b) #, c, d)