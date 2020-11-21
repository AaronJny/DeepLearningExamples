# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 21:04
# @Author  : AaronJny
# @File    : research_demo.py
# @Desc    : 研究可行性时编写的demo，效率较低
import random
import sys
from copy import deepcopy

import cv2
import numpy as np
from tqdm import tqdm

sys.setrecursionlimit(50000)


def is_marked(point):
    return sum(point) == 0


def gen_near_positions(step):
    ret = []
    for i in range(-1 * step, step + 1):
        for j in range(-1 * step, step + 1):
            if not (i == 0 and j == 0):
                ret.append((i, j))
    return ret


def check_noise_point(im, row, col, h, w):
    rate = 0.3
    total = 0
    cnt = 0
    for i, j in gen_near_positions(2):
        nx = row + i
        ny = col + j
        if 0 <= nx < h and 0 <= ny < w:
            total += 1
            if sum(im[nx][ny]) == 0:
                cnt += 1
    return cnt / total < rate


def check_normal_point(im, h, w, row, col):
    """
    判断指定点是否为正常的点
    """
    # 先获取这个点周围的所有的点的数据，按RGB分成三组数据
    dx_dy = gen_near_positions(10)
    rgb_colors = [[], [], []]
    for x, y in dx_dy:
        nx = row + x
        ny = col + y
        if 0 <= nx < h and 0 <= ny < w:
            for index, c in enumerate(im[nx][ny]):
                rgb_colors[index].append(c)
    # 按照中位数进行过滤
    for index, colors in enumerate(rgb_colors):
        seqs = np.array(colors)
        median = np.median(seqs)
        mad = np.median(np.abs(seqs - median))
        lower_limit = median - 3 * mad
        upper_limit = median + 3 * mad
        if lower_limit <= im[row][col][index] <= upper_limit:
            pass
        else:
            return False
    return True


def _dep_image(im, h, w, row, col, value, dx_dy):
    for i, j in dx_dy:
        nx = row + i
        ny = col + j
        if 0 <= nx < h and 0 <= ny < w:
            im[nx][ny] = value


def dip_image(im, h, w, step=5):
    """
    使用黑色点浸染临近像素
    """
    origin_marked_positions = []
    for i in range(h):
        for j in range(w):
            if is_marked(im[i][j]):
                origin_marked_positions.append((i, j))
    dx_dy = gen_near_positions(step)
    for row, col in origin_marked_positions:
        _dep_image(im, h, w, row, col, [0, 0, 0], dx_dy)


def search_one_domain(visited, h, w, row, col, sub_domain):
    """
    查找和给定的点联通的其他像素点，并把坐标加入到sub_domain中
    """
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]
    for x, y in zip(dx, dy):
        nx = row + x
        ny = col + y
        if h > nx >= 0 and 0 <= ny < w and visited[nx][ny] == 0:
            visited[nx][ny] = 1
            sub_domain.append((nx, ny))
            search_one_domain(visited, h, w, nx, ny, sub_domain)


def find_max_domain(im, h, w):
    """
    寻找所有连通域
    """
    # 制造一个标记用的数组
    visited = np.zeros((h, w))
    # 初始化围墙
    for row in range(h):
        for col in range(w):
            if is_marked(im[row][col]):
                visited[row][col] = 1
    print(visited)
    # 逐像素搜索连通域
    domains = []
    for row in range(h):
        for col in range(w):
            # 找没有访问过的像素
            if visited[row][col] == 0:
                # 有的话就是一个新的连通域
                sub_domain = [(row, col), ]
                visited[row][col] = 1
                search_one_domain(visited, h, w, row, col, sub_domain)
                domains.append(sub_domain)
    # 统计每个域的大小
    domains = sorted(domains, key=lambda x: -len(x))
    # 随机选择一个点返回
    position = random.choice(domains[0])
    return position


def run():
    origin_im = cv2.imread('./samples/47ae80cd-24d8-456f-9160-b39fd1949e00.jpg')
    new_im = deepcopy(origin_im)
    h, w, c = origin_im.shape
    pbar = tqdm(total=h * w * 2)
    for row in range(h):
        for col in range(w):
            if not check_normal_point(origin_im, h, w, row, col):
                new_im[row][col] = [0, 0, 0]
            pbar.update(1)
    for row in range(h):
        for col in range(w):
            if check_noise_point(new_im, row, col, h, w):
                new_im[row][col] = origin_im[row][col].copy()
            pbar.update(1)
    dip_image(new_im, h, w, 7)
    result_x, result_y = find_max_domain(new_im, h, w)
    _dep_image(new_im, h, w, result_x, result_y, [255, 0, 0], gen_near_positions(7))
    cv2.imwrite('1.jpg', new_im)


if __name__ == '__main__':
    run()
