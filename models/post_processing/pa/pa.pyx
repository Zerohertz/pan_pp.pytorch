import numpy as np
import cv2
import torch
cimport numpy as np
cimport cython
cimport libcpp
cimport libcpp.pair
cimport libcpp.queue
from libcpp.pair cimport *
from libcpp.queue cimport *

import time


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=2] _pa(np.ndarray[np.uint8_t, ndim=3] kernels,
                                        np.ndarray[np.float32_t, ndim=3] emb,
                                        np.ndarray[np.int32_t, ndim=2] label,
                                        np.ndarray[np.int32_t, ndim=2] cc,
                                        np.ndarray[np.float32_t, ndim=1] area,
                                        np.ndarray[np.npy_bool, ndim=3] inds,
                                        np.ndarray[np.int32_t, ndim=2] p,
                                        int kernel_num,
                                        int label_num,
                                        float min_area=0):
    cdef np.ndarray[np.int32_t, ndim=2] pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] mean_emb = np.zeros((label_num, 4), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] flag = np.zeros((label_num,), dtype=np.int32)

    global s1, s2, s3

    s1 = time.time()
    #####################################################################################################
    cdef np.float32_t max_rate = 1024
    cdef int i, j, tmp
    
    for i in range(1, label_num):
        ind = inds[i]
        if area[i] < min_area:
            label[ind] = 0
            continue
        for j in range(1, i):
            if area[j] < min_area:
                continue
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:
                continue
            rate = area[i] / area[j]
            if rate < 1 / max_rate or rate > max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, ind], axis=1)

                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)
    #####################################################################################################
    s1 = time.time() - s1
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    s2 = time.time()
    #####################################################################################################
    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t, np.int16_t]] que = \
        queue[libcpp.pair.pair[np.int16_t, np.int16_t]]()
    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t, np.int16_t]] nxt_que = \
        queue[libcpp.pair.pair[np.int16_t, np.int16_t]]()
    cdef np.int16_t*dx = [-1, 1, 0, 0]
    cdef np.int16_t*dy = [0, 0, -1, 1]
    cdef np.int16_t tmpx, tmpy

    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        tmpx, tmpy = points[point_idx, 0], points[point_idx, 1]
        que.push(pair[np.int16_t, np.int16_t](tmpx, tmpy))
        pred[tmpx, tmpy] = label[tmpx, tmpy]
    #####################################################################################################
    s2 = time.time() - s2
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    s3 = time.time()
    #####################################################################################################
    cdef libcpp.pair.pair[np.int16_t, np.int16_t] cur
    cdef int cur_label
    for kernel_idx in range(kernel_num - 2, -1, -1):
        while not que.empty():
            cur = que.front()
            que.pop()
            cur_label = pred[cur.first, cur.second]

            is_edge = True
            for j in range(4):
                tmpx = cur.first + dx[j]
                tmpy = cur.second + dy[j]
                if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                    continue
                if kernels[kernel_idx, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue
                if flag[cur_label] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[cur_label]) > 3:
                    continue

                que.push(pair[np.int16_t, np.int16_t](tmpx, tmpy))
                pred[tmpx, tmpy] = cur_label
                is_edge = False
            if is_edge:
                nxt_que.push(cur)

        que, nxt_que = nxt_que, que
    #####################################################################################################
    s3 = time.time() - s3
    #####################################################################################################
    return pred

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _connectedComponents(np.ndarray[np.int32_t, ndim=2] textKernel):
    cdef int H = textKernel.shape[0]
    cdef int W = textKernel.shape[1]
    cdef np.int16_t*di = [1, 0, -1, 0]
    cdef np.int16_t*dj = [0, 1, 0, -1]
    cdef np.ndarray[np.int32_t, ndim=2] label = np.zeros((H, W), dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] area = np.zeros(1, dtype=np.float32)
    cdef np.ndarray[np.npy_bool, ndim=3] inds = np.array([np.zeros((H, W), dtype=np.bool)], dtype=np.bool)
    px = [[]]
    py = [[]]
    cdef int tmp_label = 0
    cdef int i, j, tmpi, tmpj, ni, nj
    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t, np.int16_t]] que = \
        queue[libcpp.pair.pair[np.int16_t, np.int16_t]]()
    for i in range(H):
        for j in range(W):
            if textKernel[i][j] == 1:
                area = np.append(area, np.zeros(1, dtype=np.float32))
                inds = np.concatenate((inds, [np.zeros((H, W), dtype=np.bool)]))
                px.append([])
                py.append([])
                que.push(pair[np.int16_t, np.int16_t](i, j))
                textKernel[i][j] = 0
                tmp_label += 1
                while not que.empty():#
                    tmpi = que.front().first
                    tmpj = que.front().second
                    que.pop()
                    label[tmpi][tmpj] = tmp_label
                    area[tmp_label] += 1
                    inds[tmp_label][tmpi][tmpj] = True
                    px[tmp_label].append(tmpi)
                    py[tmp_label].append(tmpj)
                    for k in range(4):
                        ni = tmpi + di[k]
                        nj = tmpj + dj[k]
                        if textKernel[ni][nj] == 1 and 0 <= ni < H and 0 <= nj < W:
                            que.push(pair[np.int16_t, np.int16_t](ni, nj))
                            textKernel[ni][nj] = 0
    return label, area, inds, tmp_label + 1, px, py


def pa(kernels, emb, cc, label_num, label, area, inds, p, min_area=0):
    #####################################################################################################
    s0 = time.time()
    #####################################################################################################
    kernel_num = kernels.shape[0]
    #####################################################################################################
    s0 = time.time() - s0
    #####################################################################################################
    return _pa(kernels[:-1], emb, label, cc, area, inds, p, kernel_num, label_num, min_area), [s0, s1, s2, s3]

def connectedComponents(textKernel):
    return _connectedComponents(textKernel)