from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import open3d as o3d
from utils import distanceBetween2Points, makeCloud

# Simple utility: save-or-show behavior
def _finish_plot(save_path=None, title=None):
    if title:
        plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        plt.close()
    else:
        plt.show()

def plot3D(x, y, z, save_path=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)
    if title: ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=160)
        plt.close()
    else:
        plt.show()

def plot3DPreview(x, y, z, save_path=None, title=None, sample=5000):
    # for large clouds, plot a random subset for speed
    n = len(x)
    if n > sample:
        idx = np.random.choice(n, sample, replace=False)
        xs = [x[i] for i in idx]; ys = [y[i] for i in idx]; zs = [z[i] for i in idx]
    else:
        xs, ys, zs = x, y, z
    plot3D(xs, ys, zs, save_path=save_path, title=title)

def plot2D(x, y, save_path=None, title=None):
    plt.figure()
    plt.scatter(x, y, s=1)
    _finish_plot(save_path, title)

def plot2DWithBox(x, y, box, save_path=None, title=None):
    plt.figure()
    bl = box[0]; tr = box[2]
    plt.scatter(x, y, s=1)
    w = tr[0] - bl[0]; h = tr[1] - bl[1]
    rect = Rectangle((bl[0], bl[1]), w, h, fill=False, linewidth=2)
    plt.gca().add_patch(rect)
    _finish_plot(save_path, title)

def plot2DWithClustersCenters(x, y, centers, save_path=None, title=None):
    plt.figure()
    plt.scatter(x, y, s=1)
    if centers:
        cx = [c[0] for c in centers]
        cy = [c[1] for c in centers]
        plt.scatter(cx, cy, s=30, marker='x')
        # mark furthest center from average
        mx = sum(cx)/len(cx); my = sum(cy)/len(cy)
        plt.scatter([mx], [my], s=40, marker='o', facecolors='none')
        fur = None; md = -1
        for c in centers:
            d = distanceBetween2Points((mx,my), c)
            if d > md: md = d; fur = c
        if fur: plt.scatter([fur[0]], [fur[1]], s=60, marker='+')
    _finish_plot(save_path, title)

def plot2DWithBoxAndCenters(x, y, box, centers, save_path=None, title=None):
    plt.figure()
    bl = box[0]; tr = box[2]
    plt.scatter(x, y, s=1)
    # room box
    w = tr[0] - bl[0]; h = tr[1] - bl[1]
    rect = Rectangle((bl[0], bl[1]), w, h, fill=False, linewidth=2)
    plt.gca().add_patch(rect)
    # centers
    if centers:
        cx = [c[0] for c in centers]
        cy = [c[1] for c in centers]
        plt.scatter(cx, cy, s=40, marker='x')
    _finish_plot(save_path, title)

def plot2DWithDensity(x, y, gridsize=60, save_path=None, title=None):
    plt.figure()
    plt.hexbin(x, y, gridsize=gridsize)
    plt.colorbar(label="point density")
    _finish_plot(save_path, title)

def showCloud(x, y, z):
    cloud = makeCloud(x, y, z)
    o3d.visualization.draw_geometries([cloud])
