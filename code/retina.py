__author__ = "oalenkin"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


def cross(vec1, vec2):
    
    return T.as_tensor([
        vec1[1]*vec2[2] - vec1[2]*vec2[1],
        vec1[2]*vec2[0] - vec1[0]*vec2[2],
        vec1[0]*vec2[1] - vec1[1]*vec2[0]])

######################################################################################################################################

a0 = T.vector("a0", dtype='float64')
a = T.vector("a", dtype='float64')
b0 = T.vector("b0", dtype='float64')
b = T.vector("b", dtype='float64')

dist = T.sqrt(T.sum((a*cross((a0-b0), b))**2))/T.sqrt(T.sum(cross(a,b)**2))

distance = theano.function([a0, a, b0, b], dist)

def distance_between_skew_lines(start1, direction1, start2, direction2):
    """
    Find distance between 2 lines. Every line should be presented by 2 vectors: starting point, directing vector.
    """
    
    return distance(start1, direction1, start2, direction2)

######################################################################################################################################

points = T.matrix("points", dtype='float64')
directions = T.matrix("directions", dtype='float64')
track0 = T.vector("track0", dtype='float64')
track = T.vector("track", dtype='float64')
sig = T.scalar("sig", dtype="float64")

rs, updates = theano.scan(fn = lambda point, direction, tr0, tr, s:
                          T.exp(-(T.sqrt(T.sum((direction*cross((point-tr0), tr))**2))/T.sqrt(T.sum(cross(direction,tr))**2))**2/s**2),
                          sequences=[points, directions],
                          non_sequences=[track0, track, sig])
    
r = rs.sum()
R = theano.function([track0, track, points, directions, sig], r)

def artifitial_retina_response(track_start, track_direction, tubes_starts, tubes_directions, sigma):
    """
    Find retina response for a particular track and set of tubes.
    """
    
    return R(track_start, track_direction, tubes_starts, tubes_directions, sigma)

######################################################################################################################################

#arr = np.array(x1, y1, z1, x2, y2, z2)
def points2vec(arr):
    a0 = np.array([arr[0], arr[1], arr[2]])
    a = np.array([arr[3]-arr[0], arr[4]-arr[1], arr[5]-arr[2]])
    return a0, a

def params2vec(l, x0, m, y0):
    z1 = 0
    z2 = 1
    x1 = x0
    x2 = 1 * l + x0
    y1 = y0
    y2 = 1 * m + y0
    a0 = np.array([x1, y1, z1])
    a = np.array([x2-x1, y2-y1, z2-z1])
    return a0, a

def plot_artifitial_retina_response(event, x0s, ls, y0s, ms):
    """
    Take 4d-grid and return projections on 2 planes.
    """
    
    A0 = []
    A = []
    for i in range(len(event.index)):
        a0, a = points2vec(event[['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz']].values[i])
        A0.append(a0)
        A.append(a)
    A0 = np.array(A0)
    A = np.array(A)
    
    grid = np.ndarray(shape=(len(x0s), len(ls), len(y0s), len(ms)))
    for i in range(len(x0s)):
        for j in range(len(ls)):
            for s in range(len(y0s)):
                for t in range(len(ms)):
                    track_point, track_direction = params2vec(ls[j], x0s[i], ms[t], y0s[s])
                    grid[i, j, s, t] = artifitial_retina_response(track_point, track_direction, A0, A, 5)
    
    projection_on_yz = np.zeros((grid.shape[3], grid.shape[2]), dtype="float64")
    projection_on_xz = np.zeros((grid.shape[1], grid.shape[0]), dtype="float64")
    
    for i in range(grid.shape[1]):
        for j in range(grid.shape[0]):
            projection_on_xz[i, j] = np.max(grid[j, i, :, :])
            
    for s in range(grid.shape[3]):
        for t in range(grid.shape[2]):
            projection_on_yz[s, t] = np.max(grid[:, :, t, s])
    
    fig1 = plt.figure(figsize=(9, 7))
    plt.title("Projection on XZ", fontsize=20, fontweight='bold')
    _ = plt.imshow(projection_on_xz, aspect='auto', cmap="Blues", extent=(ls.min(), ls.max(), x0s.max(), x0s.min()))
    plt.colorbar()
    
    fig2 = plt.figure(figsize=(9, 7))
    plt.title("Projection on YZ", fontsize=20, fontweight='bold')
    _ = plt.imshow(projection_on_yz, aspect='auto', cmap="Blues", extent=(ms.min(), ms.max(), y0s.max(), y0s.min()))
    plt.colorbar()
            
    return fig1, fig2