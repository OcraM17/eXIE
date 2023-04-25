import os.path
import numpy as np
import torchvision
import torch
import argparse
from Actions import select
from kornia.losses import psnr_loss

ACTIONS = 26


def parse_args():
    parser = argparse.ArgumentParser("eXIE Parser")
    a = parser.add_argument
    a("-m", help="Dijkstra,Best Greedy, eXIE")
    a("-t", type=float, help="SOGLIA")
    a("-s", type=int, help="STEPS")
    a("--basedir", type=str, help="BASEDIR")
    a("--expname", type=str, help="EXPNAME")
    return parser.parse_args()


class Node:
    def __init__(self, img, idx):
        self.img = img
        self.idx = idx


class Tree:
    def __init__(self, img, target):
        self.root = Node(img, 0)
        self.target = target
        self.build_tree()

    def build_tree(self):
        self.graph = [self.root]


def apply_act(img, idx):
    val = torch.from_numpy(img).permute([2, 0, 1])
    val = select(val.unsqueeze(0), idx).squeeze(0)
    val = val.permute([1, 2, 0]).numpy()
    return val


class eXIE:
    def __init__(self, graph, target, mode, THR, steps):
        self.graph = graph
        self.g = np.full(1, np.inf)
        self.h = np.full(1, np.inf)
        self.c = np.full((1, ACTIONS), -1)
        self.target = target
        if mode == 'Dijkstra':
            self.mode = (0, 1)
        if mode == 'Greedy':
            self.mode = (1, 0)
        if mode == 'eXIE':
            self.mode = (1, 1)
        self.update_f()
        self.thr = THR
        self.steps = steps

    def compute_contrast(self, img1, img2):
        k = 0
        m = 100
        while True:
            diff = np.abs(np.std(img2) - np.std(img1))
            if diff < m:
                k += 1
                m = diff
                img1 = apply_act(img1, 12)
            else:
                break
        return k

    def compute_gammas(self, img1, img2):
        k = 0
        m = 100
        diff = np.mean(img2) - np.mean(img1)
        factor = 0.6 if diff > 0 else 1.1
        while True:
            diff = np.abs(np.mean(img2) - np.mean(img1))
            if diff < m:
                k += 1
                m = diff
                img1 = img1 ** factor
            else:
                break
        return k

    def pixel_step_gamma(self, x, y):
        z = np.full(x.shape, np.inf)
        x, y = np.clip(x, 1e-8, 0.999), np.clip(y, 1e-8, 0.999)
        z[x > y] = np.floor(np.log(np.log(y[x > y]) / np.log(x[x > y])) / np.log(1.1))
        z[x < y] = np.floor(np.log(np.log(y[x < y]) / np.log(x[x < y])) / np.log(0.6))
        return z

    def pixel_step_contrast(self, x, y):
        z = np.full(x.shape, np.inf)
        cond1 = np.logical_and(x > 0.5, y > 0.5)
        cond2 = np.logical_and(x < 0.5, y < 0.5)
        x, y = np.clip(x, 1e-8, 0.999), np.clip(y, 1e-8, 0.999)
        z[cond1] = np.floor(np.log((y[cond1] - 0.5) / (x[cond1] - 0.5)) / np.log(1.414))
        z[cond2] = np.floor(np.log((y[cond2] - 0.5) / (x[cond2] - 0.5)) / np.log(0.894))
        return z

    def compute_h(self, node):
        step_bright = np.floor(np.abs(self.target - node.img) / 0.005)
        step_contrast = self.pixel_step_contrast(node.img, self.target)
        step_gamma = self.pixel_step_gamma(node.img, self.target)
        k = np.stack([step_bright, step_contrast, step_gamma], 0)
        k = np.min(k, 0)
        return k.max()

    def criteria(self, node):
        return np.sqrt(np.sum((node.img - self.target) ** 2))

    def recover_path(self, vbest):
        l = []
        l.append(vbest)
        parent = self.path[vbest]
        while parent is not None:
            l.append(parent)
            vbest = parent
            parent = self.path[vbest]
        l = l[::-1]
        s = [(l[c], (l[c + 1] - 1) % ACTIONS) for c in range(len(l) - 1)]
        return s

    def update_f(self):
        self.f = self.h * self.mode[0] + self.g * self.mode[1]

    def play(self):
        open = [self.graph[0].idx]
        closed = []
        self.g[self.graph[0].idx] = 0
        self.h[self.graph[0].idx] = self.compute_h(self.graph[0])
        self.update_f()
        self.path = {0: None}
        count = 0
        while open:
            vbest = open[np.argmin(self.f[open])]
            open.remove(vbest)
            count += 1
            if count == self.steps:
                li = {j: self.criteria(self.graph[j]) for j in closed}
                vbest = min(li, key=li.get)
                return 0, self.recover_path(vbest)
            closed.append(vbest)
            if self.criteria(self.graph[vbest]) < self.thr:
                return 1, self.recover_path(vbest)
            if np.all(self.c[vbest] == -1):
                offset = self.c.shape[0]
                self.c[vbest] = np.arange(offset, offset + ACTIONS)
                self.c = np.concatenate([self.c, np.full((ACTIONS, ACTIONS), -1)])
                self.graph += [Node(apply_act(self.graph[vbest].img, i), offset + i) for i in range(ACTIONS)]
                self.g = np.concatenate([self.g, np.full(ACTIONS, np.inf)])
                self.h = np.concatenate([self.h, np.full(ACTIONS, -1)])
            for son in self.c[vbest]:
                value = self.g[vbest] + 1
                if son in closed:
                    if value < self.g[vbest]:
                        closed.remove(son)
                        open.append(son)
                    else:
                        continue
                else:
                    if son not in open:
                        open.append(son)
                    else:
                        if value >= self.g[son]:
                            continue
                self.g[son] = value
                self.h[son] = self.compute_h(self.graph[son])
                self.update_f()
                self.path[son] = vbest


if __name__ == '__main__':
    args = parse_args()
    BASEDIR = args.basedir
    FILE = os.path.join(BASEDIR, 'test-list.txt')
    f = open(FILE)
    results_file = open(os.path.join(BASEDIR, args.expname), 'w')
    p = []
    for counter, line in enumerate(f):
        img_name = line.strip().split('.')[0]
        x = torchvision.transforms.Resize((32, 32))(
            torchvision.io.read_image(os.path.join(BASEDIR, 'raw', img_name + '.png').float() / 255.0))
        y = torchvision.transforms.Resize((32, 32))(
            torchvision.io.read_image(os.path.join(BASEDIR, 'method_results', img_name + '.png')).float() / 255.0)
        y = y.permute([1, 2, 0]).numpy()
        x = x.permute([1, 2, 0]).numpy()
        tree = Tree(x, y)
        alg = eXIE(tree.graph, tree.target, args.m, args.t, args.s)
        out, acts = alg.play()
        x = torchvision.io.read_image(os.path.join(BASEDIR, 'raw', img_name + '.png').float() / 255.0).unsqueeze(0)
        gt = torchvision.io.read_image(os.path.join(BASEDIR, 'expC', img_name + '.png').float() / 255.0).unsqueeze(0)
        for s in acts:
            x = select(x, s[1])
        p.append(-psnr_loss(x, gt, 1.0).item())
        torchvision.io.write_png((x.squeeze(0) * 255.0).to(torch.uint8), os.path.join(
            BASEDIR, 'eXIE_results', img_name + '.png'))
        results_file.write(img_name + ' ' + ' '.join(map(str, acts)) + '\n')
