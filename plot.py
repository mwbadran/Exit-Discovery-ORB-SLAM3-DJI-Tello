from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import open3d as o3d
from utils import distanceBetween2Points, makeCloud

def plot3D(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)
    plt.show()

def plot2D(x, y):
    plt.scatter(x, y, s=1)
    plt.axis('equal')
    plt.show()

def plot2DWithBox(x, y, box):
    bl = box[0]; tr = box[2]
    plt.scatter(x, y, s=1)
    w = tr[0] - bl[0]; h = tr[1] - bl[1]
    rect = Rectangle((bl[0], bl[1]), w, h, fill=False, linewidth=2)
    plt.gca().add_patch(rect)
    plt.axis('equal')
    plt.show()

def plot2DWithClustersCenters(x, y, centers):
    plt.scatter(x, y, s=1)
    if centers:
        cx = [c[0] for c in centers]
        cy = [c[1] for c in centers]
        plt.scatter(cx, cy, s=20)
        # mark furthest center from average
        mx = sum(cx)/len(cx); my = sum(cy)/len(cy)
        plt.scatter([mx], [my], s=30)
        fur = None; md = -1
        for c in centers:
            d = distanceBetween2Points((mx,my), c)
            if d > md: md = d; fur = c
        if fur: plt.scatter([fur[0]], [fur[1]], s=30)
    plt.axis('equal')
    plt.show()

def showCloud(x, y, z):
    cloud = makeCloud(x, y, z)
    o3d.visualization.draw_geometries([cloud])
