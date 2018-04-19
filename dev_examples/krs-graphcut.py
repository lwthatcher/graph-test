#! /usr/bin/env python3

import cv2 as cv3
import graph_tool, graph_tool.generation as gt
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, product
import argparse
from pdb import set_trace


class KernalDensityEstimator:
    sigf = 50.0
    minv2sigf2 = -1.0 / (2.0 * sigf ** 2)
    sigb = 50.0
    minv2sigb2 = -1.0 / (2.0 * sigb ** 2)

    def __init__(self, fs, bs):
        self.fs = np.array(fs, dtype='float')
        self.bs = np.array(bs, dtype='float')

    def probs(self, color):
        pf = np.mean(np.exp(np.power(self.fs - color, 2) * self.minv2sigf2))
        pb = np.mean(np.exp(np.power(self.bs - color, 2) * self.minv2sigb2))
        if pf == 0.0 and pb == 0.0:
            return 0.5, 0.5
        else:
            return pf / (pf + pb), pb / (pf + pb)
            # return -np.log2(pf), -np.log2(pb)


def main(filename, f_seeds, b_seeds):
    # Load image
    print("Reading", filename, flush=True, end="...")
    img = cv3.imread(filename, cv3.IMREAD_COLOR)[:, :, ::-1]
    rows, cols, _ = img.shape
    print("Done")

    # Init graph
    print("Initializing graph", flush=True, end="...")
    g = gt.lattice((rows, cols))
    v_r = g.new_vertex_property('int')
    v_g = g.new_vertex_property('int')
    v_b = g.new_vertex_property('int')
    e_w = g.new_edge_property('float')
    print("Done")

    # Make a row,col map of the indices for easy access
    dex = np.zeros((rows, cols), dtype='int')
    np.ravel(dex)[:] = range(rows * cols)

    # Load vertex property arrays with pixel values
    print("Loading pixel values", flush=True, end="...")
    v_r.a = np.ravel(img[:, :, 0])
    v_g.a = np.ravel(img[:, :, 1])
    v_b.a = np.ravel(img[:, :, 2])
    print("Done")

    # Weight n-link edges
    print("Setting n-link weights", flush=True, end="...")

    def n_weight(s_r, s_g, s_b, t_r, t_g, t_b):
        return 1.0 - 3.0 ** -0.5 * ((v_r[s] - v_r[t]) ** 2 + (v_g[s] - v_g[t]) ** 2 +
                                    (v_b[s] - v_b[t]) ** 2) / 65536.0  # Scale to 1

    for e in g.edges():
        s, t = e.source(), e.target()
        e_w[e] = 0.13 * n_weight(v_r[s], v_g[s], v_b[s], v_r[t], v_g[t], v_b[t])
    print("Done")

    # Double up and make directed
    print("Converting to directed graph", flush=True, end="...")
    g.set_directed(True)
    g.add_edge_list([(e.target(), e.source(), e_w[e]) for e in g.edges()],
                    eprops=(e_w,))
    print("Done")

    # Make t-link edges
    print("Making t-link edges", flush=True, end="...")
    f, b = g.add_vertex(2)
    for n in range(rows * cols):
        g.add_edge(f, n)
        g.add_edge(n, b)
    print("Done")

    # Seeds
    kde = KernalDensityEstimator([img[c] for c in f_seeds],
                                 [img[c] for c in b_seeds])

    # Weight t-link edges
    print("Setting t-link weights", flush=True, end="...")
    for row, col in product(range(rows), range(cols)):
        if (row, col) in f_seeds:
            e_w[g.edge(f, g.vertex(dex[row, col]))] = 1e9
            e_w[g.edge(g.vertex(dex[row, col]), b)] = 0.0
        elif (row, col) in b_seeds:
            e_w[g.edge(f, g.vertex(dex[row, col]))] = 0.0
            e_w[g.edge(g.vertex(dex[row, col]), b)] = 1e9
        else:
            (e_w[g.edge(f, g.vertex(dex[row, col]))],
             e_w[g.edge(g.vertex(dex[row, col]), b)]) = kde.probs(img[row, col])
    print("Done")

    # Perform the cut
    print("Making the cut", flush=True, end="...")
    res = gt.flow.boykov_kolmogorov_max_flow(g, f, b, e_w)
    cut = gt.flow.min_st_cut(g, f, e_w, res)
    print("Done")

    # Display the result
    print("Highlighting", flush=True, end="...")
    highlight = np.array((0.2, 0.8, 0.8))
    for row, col in product(range(rows), range(cols)):
        if not cut[g.vertex(dex[row, col])]:
            img[row, col] = 255 - highlight * (255 - img[row, col])
            # img[row, col] = (0, 0, 128)
    print("Done")

    display_image(img)

    return g, e_w, v_r, cut


def greg():
    f_seeds = [(32, 22), (20, 32), (30, 27), (30, 35), (35, 32), (32, 42),
               (11, 32), (15, 32), (55, 28), (55, 36)]
    b_seeds = [(10, 10), (30, 10), (50, 10), (10, 50), (30, 50), (50, 50),
               (55, 24), (55, 42), (60, 5)]
    return main('Greg64.jpg', f_seeds, b_seeds)


def stonehenge():
    f_seeds = [(y, x) for x, y in product(range(60, 101, 20), range(40, 240, 5))]
    b_seeds = [(y, x) for x, y in product(range(160, 301, 20), range(40, 240, 20))]
    print(len(f_seeds))
    print(len(b_seeds))
    return main('stonehenge.jpg', f_seeds, b_seeds)


def berry():
    f_seeds = [(y, 144) for y in range(102, 145)] + \
              [(130, x) for x in range(140, 200)]
    b_seeds = [(64, x) for x in range(32, 320, 32)] + \
              [(192, x) for x in range(32, 320, 32)]
    return main('IMG_3186.jpg', f_seeds, b_seeds)


def arkenstone():
    f_seeds = [(y, 180) for y in range(86, 148, 2)] + \
              [(119, x) for x in range(152, 212, 2)]
    b_seeds = [(60, 60), (200, 280), (60, 280), (200, 60)]
    return main('IMG_6254.jpg', f_seeds, b_seeds)


def display_image(image, wait=1):
    plt.clf()
    plt.imshow(image, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.ion()
    plt.draw()
    if wait > 0:
        plt.pause(wait)
    else:
        plt.pause(0.001)
        input('Press [enter] to continue.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GrabCut with naive kde")
    parser.add_argument('image', choices=['greg', 'stonehenge', 'berry', 'arkenstone'])
    args = parser.parse_args()

    if args.image == 'greg':
        greg()
    elif args.image == 'stonehenge':
        stonehenge()
    elif args.image == 'berry':
        berry()
    elif args.image == 'arkenstone':
        arkenstone()