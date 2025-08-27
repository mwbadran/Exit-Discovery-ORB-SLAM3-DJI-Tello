from math import log2, floor
from utils import distanceBetween2Points

def getAverageRectangle(x, y):
    cx = float(sum(x) / len(x))
    cy = float(sum(y) / len(y))
    center = (cx, cy)

    leftX  = [(x[i], y[i]) for i in range(len(x)) if x[i] <  cx]
    rightX = [(x[i], y[i]) for i in range(len(x)) if x[i] >  cx]
    upY    = [(x[i], y[i]) for i in range(len(y)) if y[i] >  cy]
    downY  = [(x[i], y[i]) for i in range(len(y)) if y[i] <  cy]

    ld = [distanceBetween2Points(p, center) for p in leftX]  or [0.0]
    rd = [distanceBetween2Points(p, center) for p in rightX] or [0.0]
    ud = [distanceBetween2Points(p, center) for p in upY]    or [0.0]
    dd = [distanceBetween2Points(p, center) for p in downY]  or [0.0]

    xLeft  = cx - (2 * float(sum(ld) / len(ld)))
    xRight = cx + (2 * float(sum(rd) / len(rd)))
    yUp    = cy + (2 * float(sum(ud) / len(ud)))
    yDown  = cy - (2 * float(sum(dd) / len(dd)))

    return (xLeft, yDown), (xRight, yDown), (xRight, yUp), (xLeft, yUp)

# The rest (findBestBoundingBox, etc.) is kept for reference but unused.
def splitIntoSubarrays(array, k):
    start = 0; subs = []
    if k <= 0: k = 1
    while start <= len(array):
        end = min(start + k, len(array))
        if start == end: break
        subs.append(array[start:end]); start += k
    return subs

def boundingBox(points):
    mnx, mny = float('inf'), float('inf')
    mxx, mxy = float('-inf'), float('-inf')
    for x, _, y in points:
        mnx = min(mnx, x); mny = min(mny, y)
        mxx = max(mxx, x); mxy = max(mxy, y)
    return (mnx, mny), (mxx, mny), (mxx, mxy), (mnx, mxy)

def getBoxFitness(box, points):
    bl = box[0]; tr = box[2]
    fit = 0
    for p in points:
        on_tb = bl[0] <= p[0] <= tr[0] and (p[1] == bl[1] or p[1] == tr[1])
        on_lr = bl[1] <= p[1] <= tr[1] and (p[0] == tr[0] or p[0] == bl[0])
        if on_tb or on_lr: continue
        inside = (bl[0] < p[0] < tr[0]) and (bl[1] < p[1] < tr[1])
        fit += 1 if inside else -1
    return abs(fit)
