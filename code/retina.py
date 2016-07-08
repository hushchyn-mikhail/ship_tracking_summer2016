__author__ = "oalenkin"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from scipy.optimize import minimize_scalar


#arr = np.array(x1, y1, z1, x2, y2, z2)
def points2vec(arr):
    a0 = np.array([arr[0], arr[1], arr[2]])
    a = np.array([arr[3]-arr[0], arr[4]-arr[1], arr[5]-arr[2]])
    return a0, a

def params2vec(x0, l, y0, m):
    z1 = 0
    z2 = 1
    x1 = x0
    x2 = 1 * l + x0
    y1 = y0
    y2 = 1 * m + y0
    a0 = np.array([x1, y1, z1])
    a = np.array([x2-x1, y2-y1, z2-z1])
    return a0, a

def vec2params(a0, a):
    
    l = a[0] / a[2]
    m = a[1] / a[2]
    x0 = a0[0] - a[0] * a0[2] / a[2]
    y0 = a0[1] - a[1] * a0[2] / a[2]
    return x0, l, y0, m

######################################################################################################################################

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
R_func = theano.function([track0, track, points, directions, sig], r)

derivative_r = T.grad(r, [track0, track])
derivative_R = theano.function([track0, track, points, directions, sig], derivative_r)

def artifitial_retina_response(l, x0, m, y0, coordinates, sigma):
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
    
    track_start, track_direction = params2vec(x0, l, y0, m)
    
    tubes_starts = []
    tubes_directions = []
    for i in range(len(coordinates)):
        a0, a = points2vec(coordinates[i])
        tubes_starts.append(a0)
        tubes_directions.append(a)
    tubes_starts = np.array(tubes_starts)
    tubes_directions = np.array(tubes_directions)
    
    return R_func(track_start, track_direction, tubes_starts, tubes_directions, sigma)

######################################################################################################################################

def plot_artifitial_retina_response(event, ms, y0s, ls, x0s, sigma):
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
    
    coordinates = event[['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz']].values
    
    #4d-grid of responses
    grid = np.ndarray(shape=(len(x0s), len(ls), len(y0s), len(ms)))
    for i in range(len(x0s)):
        for j in range(len(ls)):
            for s in range(len(y0s)):
                for t in range(len(ms)):
                    grid[i, j, s, t] = artifitial_retina_response(ls[j], x0s[i], ms[t], y0s[s], coordinates, sigma)
    
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
    #_ = plt.imshow(projection_on_xz, aspect='auto', cmap="Blues", extent=(ls.min(), ls.max(), x0s.max(), x0s.min()))
    _ = plt.contourf(x0s, ls, projection_on_xz, cmap=plt.cm.Oranges)
    plt.xlabel("x0")
    plt.ylabel("l")
    plt.colorbar()
    
    fig2 = plt.figure(figsize=(9, 7))
    plt.title("Projection on YZ", fontsize=20, fontweight='bold')
    #_ = plt.imshow(projection_on_yz, aspect='auto', cmap="Blues", extent=(ms.min(), ms.max(), y0s.max(), y0s.min()))
    _ = plt.contourf(y0s, ms, projection_on_yz, cmap=plt.cm.Oranges)
    plt.xlabel("y0")
    plt.ylabel("m")
    plt.colorbar()
            
    return fig1, fig2

######################################################################################################################################

def get_track_params(event, trackID):
    """
    Returns x0, l, y0, m parameters of track.
    Input:
        event - pandas dataframe containing all hits of any event before/after magnet;
        trackID - id of track.
    Output:
        x0, l, y0, m - parametres of track.
    """
    track = event[event.TrackID==trackID]
    Xs = track.X.values
    Ys = track.Y.values
    Zs = track.Z.values
    
    x_params = np.polyfit(Zs[:2], Xs[:2], 1)
    x0 = x_params[1]
    l = x_params[0]

    y_params = np.polyfit(Zs[:2], Ys[:2], 1)
    y0 = y_params[1]
    m = y_params[0]
    
    return x0, l, y0, m

######################################################################################################################################

def grad_step(dot):
    
    return minimize(dot)

class RetinaTrackReconstruction(object):
    
    def __init__(self):

        self.labels_ = None
        self.tracks_params_ = None
        
    def grad_R(self, a):
        
        return -np.array(derivative_R(a[0], a[1], self.tubes_starts, self.tubes_directions, self.sigma))
    
    def R(self, a):
        
        return -R_func(a[0], a[1], self.tubes_starts, self.tubes_directions, self.sigma)
    
    def minimize(self, a):
    
        l_min = minimize_scalar(lambda l: self.R(a - l * self.grad_R(a))).x
        return a - l_min * self.grad_R(a)
    
    def grad_step(self, a):
        
        return self.minimize(a)
    
    def gradient_descent(self, initial_dot, eps):
        
        dots = [initial_dot]
        values = [self.R(dots[-1])]
        dots.append(self.grad_step(dots[-1]))
        values.append(self.R(dots[-1]))        
        
        while np.linalg.norm(dots[-2]-dots[-1])>eps:
            dots.append(self.grad_step(dots[-1]))
            values.append(self.R(dots[-1]))  
            
        dots = np.array(dots)
        values = np.array(values)
            
        return dots, values

    def fit(self, ends_of_strawtubes, ipar, s, eps=0.000005):
        
        A0 = []
        A = []
        
        for i in range(len(ends_of_strawtubes)):
            
            a0, a = points2vec(ends_of_strawtubes[i])
            A0.append(a0)
            A.append(a)
        
        self.tubes_starts = np.array(A0)
        self.tubes_directions = np.array(A)
        self.sigma = s
        
        initial_track_start, initial_track_direction = params2vec(ipar[0], ipar[1], ipar[2], ipar[3])
        start_dot = np.array([initial_track_start, initial_track_direction])
        
        dots, values = self.gradient_descent(start_dot, eps)

        return vec2params(dots[-1][0], dots[-1][1]), dots, values
        #self.labels_ = labels
        #self.tracks_params_ = tracks_params