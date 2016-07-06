__author__ = "oalenkin"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


def cross(vec1, vec2):
    """
    Cross product.
    """
    
    return T.as_tensor([
        vec1[1]*vec2[2] - vec1[2]*vec2[1],
        vec1[2]*vec2[0] - vec1[0]*vec2[2],
        vec1[0]*vec2[1] - vec1[1]*vec2[0]])

######################################################################################################################################

a0 = T.vector("a0", dtype='float64')
a = T.vector("a", dtype='float64')
b0 = T.vector("b0", dtype='float64')
b = T.vector("b", dtype='float64')

#theano realization of distance function: ||(m,[u,v])|| / ||[u,v]||, where u, v - directing vectors, m connects any points of these lines
dist = T.sqrt(T.sum((a*cross((a0-b0), b))**2))/T.sqrt(T.sum(cross(a,b)**2))

distance = theano.function([a0, a, b0, b], dist)

def distance_between_skew_lines(start1, direction1, start2, direction2):
    """
    Find distance between 2 lines. Every line should be presented by 2 vectors: starting point, directing vector.
    Input:
        start1 - vector of any point on the first line;
        direction1 - directing vector of the first line;
        start2 - vector of any point on the second line;
        direction2 - directing vector of the second line.
    Output:
        distance - distance between lines.
    """
    
    return distance(start1, direction1, start2, direction2)

######################################################################################################################################

points = T.matrix("points", dtype='float64')
directions = T.matrix("directions", dtype='float64')
track0 = T.vector("track0", dtype='float64')
track = T.vector("track", dtype='float64')
sig = T.scalar("sig", dtype="float64")

#theano realization of retina_response function: sum(exp(-dist**2/sigma**2))), where dist() is distance_between_skew_lines()
rs, updates = theano.scan(fn = lambda point, direction, tr0, tr, s:
                          T.exp(-(T.sqrt(T.sum((direction*cross((point-tr0), tr))**2))/T.sqrt(T.sum(cross(direction,tr)**2)))**2/s**2),
                          sequences=[points, directions],
                          non_sequences=[track0, track, sig])
    
r = rs.sum()
R = theano.function([track0, track, points, directions, sig], r)

def artifitial_retina_response(track_start, track_direction, coordinates, sigma):
    """
    Find retina response for a particular track and set of tubes.
    Input:
        track_start - vector of any point of track;
        track_direction - directing vector of track;
        coordinates - matrix containing coordinates of ends of tubes, one row = ['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz'];
        sigma - constant.
    Output:
        R - retina response for event.
    """
    
    tubes_starts = []
    tubes_directions = []
    for i in range(len(coordinates)):
        a0, a = points2vec(coordinates[i])
        tubes_starts.append(a0)
        tubes_directions.append(a)
    tubes_starts = np.array(tubes_starts)
    tubes_directions = np.array(tubes_directions)
    
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

def plot_artifitial_retina_response(grid, ms, y0s, ls, x0s):
    """
    Create projections on XZ and YZ.
    (x-x0)/l=(y-y0)/m=z (*)
    Input:
        grid - 4d-grid with retina responses for different combinations of parameters of track;
        x0s, ls, yos, ms - arrays with values of parameters from parametric equation(*) of line.
    Output:
        fig1 - projection of artifitial_retina_response on XZ;
        fig2 - projection of artifitial_retina_response on YZ.
    """
    
    #initialization of projection matrixes
    projection_on_yz = np.zeros((grid.shape[3], grid.shape[2]), dtype="float64")
    projection_on_xz = np.zeros((grid.shape[1], grid.shape[0]), dtype="float64")
    
    #filling matrixes
    for i in range(grid.shape[1]):
        for j in range(grid.shape[0]):
            projection_on_xz[i, j] = np.max(grid[j, i, :, :])
            
    for s in range(grid.shape[3]):
        for t in range(grid.shape[2]):
            projection_on_yz[s, t] = np.max(grid[:, :, t, s])
    
    #creating of plt objects
    fig1 = plt.figure(figsize=(9, 7))
    plt.title("Projection on XZ", fontsize=20, fontweight='bold')
    _ = plt.imshow(projection_on_xz, aspect='auto', cmap="Blues", extent=(ls.min(), ls.max(), x0s.max(), x0s.min()))
    plt.colorbar()
    
    fig2 = plt.figure(figsize=(9, 7))
    plt.title("Projection on YZ", fontsize=20, fontweight='bold')
    _ = plt.imshow(projection_on_yz, aspect='auto', cmap="Blues", extent=(ms.min(), ms.max(), y0s.max(), y0s.min()))
    plt.colorbar()
            
    return fig1, fig2