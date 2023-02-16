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

# from cython.parallel import prange

import time
import csv


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=2] _boxgen(np.ndarray[np.int32_t, ndim=2] label,
                                        np.ndarray[np.float32_t, ndim=2] score,
                                        int label_num,
                                        float min_area,
                                        float min_score,
                                        np.ndarray[np.float32_t, ndim=1] scale,
                                        float pos_const,
                                        float len_const):
    cdef int W, H
    W = label.shape[0]
    H = label.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=3] inds = np.zeros((label_num, W, H), dtype=np.bool)
    cdef np.ndarray[np.int32_t, ndim=1] area = np.full(label_num, -1, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] score_i = np.full(label_num, -1, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] points = np.full((W*H, 3), label_num + 1, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] points_idx = np.zeros(W*H, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] pos = np.zeros(2, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] length = np.zeros(2, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=3] bboxes = np.zeros((0, 4, 2), dtype=np.int32)
    cdef int i, j, tmp
    cdef float deg
    cdef tuple pos_t, length_t ###
    
    for i in range(W):
        for j in range(H):
            tmp = label[i][j]
            if tmp == 0:
                continue
            else:
                inds[tmp][i][j] = True
                if area[tmp] == -1:
                    area[tmp] = 1
                    score_i[tmp] = score[i][j]
                else:
                    area[tmp] += 1
                    score_i[tmp] += score[i][j]
            points[i+W*j] = np.array((tmp, i, j), dtype=np.int32)
    points_idx = np.argsort(points, axis=0)[:, 0].astype('int32')
    points = points[points_idx][:, 1:3]
    tmp = 0
    for i in range(1, label_num):
        if area[i] < min_area:
            tmp += area[i]
            label[inds[i]] = 0
            continue
            
#         score_i = np.mean(score[inds[i]])
        if score_i[i] / area[i] < min_score:
            tmp += area[i]
            label[inds[i]] = 0
            continue
        
        pos_t, length_t, deg = cv2.minAreaRect(points[tmp:tmp+area[i]][:, ::-1])
#         if 45 < deg <= 135:
#             deg = 90
#         elif -45 <= deg <= 45:
#             deg = 0
#         if 90 - deg_const <= deg < 91:
#             deg = (deg + 90) / 2
#         elif -1 < deg <= deg_const:
#             deg = (deg) / 2
        pos, length = np.array(pos_t, dtype=np.float32), np.array(length_t, dtype=np.float32)
        pos += pos_const
        length += len_const
        pos, length = pos*scale, length*scale
        bbox = cv2.boxPoints((pos, length, deg))
        bboxes = np.append(bboxes, bbox.astype('int32').reshape(1, 4, 2), axis=0)
        tmp += area[i]
    return bboxes

def boxgen(label, score, label_num, min_area, min_score, scale, pos_const, len_const):
    return _boxgen(label, score, label_num, min_area, min_score, scale, pos_const, len_const)