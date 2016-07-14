__author__ = "oalenkin"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import MinMaxScaler


######################################################################################################################################

param_vec = T.vector("param_vec", dtype='float64')
matrix_conv = T.matrix("matrix_conv", dtype='float64')
tube_vec0 = T.vector("tube_vec0", dtype='float64')
tube_vec1 = T.vector("tube_vec1", dtype='float64')

point_in_z = T.dot(matrix_conv, param_vec)
z_dist = T.sqrt(T.sum(-T.sum((tube_vec0-point_in_z)*tube_vec1)*tube_vec1+(tube_vec0-point_in_z))**2)
z_distance_theano = theano.function([param_vec, matrix_conv, tube_vec0, tube_vec1], z_dist)
z_distance_theano_grad = theano.function([param_vec, matrix_conv, tube_vec0, tube_vec1], T.grad(z_dist, param_vec))

def z_distance(param_vec, z, tube_vec0, tube_vec1):
    
    matrix_conv = np.array([[1,z,0,0],[0,0,1,z]])
    
    return z_distance_theano(param_vec, matrix_conv, tube_vec0, tube_vec1)

def z_distance_grad(param_vec, z, tube_vec0, tube_vec1):
    
    matrix_conv = np.array([[1,z,0,0],[0,0,1,z]])
    
    return z_distance_theano_grad(param_vec, matrix_conv, tube_vec0, tube_vec1)

######################################################################################################################################

def ends2params(array):
    
    start = array[:2]
    k = (array[4]-array[1])/(array[3]-array[0])
    direction = np.array([1., k])/np.sqrt(1+k**2)
    z0 = array[5]
    
    return start, direction, z0

def artifitial_retina_response(track_params, ends_of_strawtubes, sigma):
    
    R = 0
    
    for i in range(len(ends_of_strawtubes)):
        
        start, direction, z0 = ends2params(ends_of_strawtubes[i])
        R += np.exp(-z_distance(track_params, z0, start, direction)**2/sigma**2)
        
    return R

######################################################################################################################################

def artifitial_retina_response_grad(track_params, ends_of_strawtubes, sigma):
    
    grad_R = 0
    
    for i in range(len(ends_of_strawtubes)):
    
        start, direction, z0 = ends2params(ends_of_strawtubes[i])
        rho = z_distance(track_params, z0, start, direction)
        grad_rho = z_distance_grad(track_params, z0, start, direction)
        grad_R += np.exp(-rho**2/sigma**2)*rho*grad_rho
    
    grad_R = -1./sigma**2 * grad_R
    
    return grad_R

######################################################################################################################################

def get_track_params(event, trackID):
    """
    Returns x0, l, y0, m parameters of track.
    Input:
        event - pandas dataframe containing all hits of any event before/after magnet;
        trackID - id of track.
    Output:
        [x0, l, y0, m] - list of parametres of track.
    """
    track = event[event.TrackID==trackID]
    Xs = track.X.values
    Ys = track.Y.values
    Zs = track.Z.values
    
    x_params = np.polyfit(Zs, Xs, 1)
    x0 = x_params[1]
    l = x_params[0]

    y_params = np.polyfit(Zs, Ys, 1)
    y0 = y_params[1]
    m = y_params[0]
    
    return [x0, l, y0, m]

######################################################################################################################################

def plot_artifitial_retina_response(event, params_array, sigma):
    """
    Create projections on XZ and YZ.
    (x-x0)/l=(y-y0)/m=z (*)
    Input:
        event - numpy table of event containig columns 'Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2';
        params_array - arrays with values of parameters from parametric equation(*) of line, one row is [x0, l, y0, m];
        sigma - parameter of artifitial_retina_response().
    Output:
        fig1 - projection of artifitial_retina_response on XZ;
        fig2 - projection of artifitial_retina_response on YZ.
    """

    x0s = params_array[0]
    ls = params_array[1]
    y0s = params_array[2]
    ms = params_array[3]
    
    coordinates = event[['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz']].values
    
    #4d-grid of responses
    grid = np.ndarray(shape=(len(x0s), len(ls), len(y0s), len(ms)))
    for i in range(len(x0s)):
        for j in range(len(ls)):
            for s in range(len(y0s)):
                for t in range(len(ms)):
                    grid[i, j, s, t] = artifitial_retina_response([x0s[i], ls[j], y0s[s], ms[t]], coordinates, sigma)
    
    #initialization of projection matrixes
    projection_on_yz = np.zeros((grid.shape[3], grid.shape[2]), dtype="float64")
    projection_on_xz = np.zeros((grid.shape[1], grid.shape[0]), dtype="float64")
    
    #filling matrixes
    for s in range(grid.shape[3]):
        for t in range(grid.shape[2]):
            projection_on_yz[s, t] = np.max(grid[:, :, t, s])
    
    for i in range(grid.shape[1]):
        for j in range(grid.shape[0]):
            projection_on_xz[i, j] = np.max(grid[j, i, :, :])
            
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

class Scaler():
    
    def __init__(self):
        
        self.x_range = [-1, 1]
        self.y_range = [-1, 1]
        self.z_range = [0, 2]
        
        self.x_scaler = MinMaxScaler(self.x_range)
        self.y_scaler = MinMaxScaler(self.y_range)
        self.z_scaler = MinMaxScaler(self.z_range)
        
        self.x_range_ext = [-250, 250]
        self.y_range_ext = [-500, 500]
        self.z_range_ext = [0, 3500]
        
        self.x_scaler.fit(self.x_range_ext)
        self.y_scaler.fit(self.y_range_ext)
        self.z_scaler.fit(self.z_range_ext)
        
        self.x_coeff = 1. * (self.x_range_ext[1] - self.x_range_ext[0]) / (self.x_range[1] - self.x_range[0])
        self.y_coeff = 1. * (self.y_range_ext[1] - self.y_range_ext[0]) / (self.y_range[1] - self.y_range[0])
        self.z_coeff = 1. * (self.z_range_ext[1] - self.z_range_ext[0]) / (self.z_range[1] - self.z_range[0])
        
    def transform(self, coordinates):
        
        new_coordinates = np.copy(coordinates)
        new_coordinates[:, 0] = self.x_scaler.transform(coordinates[:, 0])
        new_coordinates[:, 3] = self.x_scaler.transform(coordinates[:, 3])
        new_coordinates[:, 1] = self.y_scaler.transform(coordinates[:, 1])
        new_coordinates[:, 4] = self.y_scaler.transform(coordinates[:, 4])
        new_coordinates[:, 2] = self.z_scaler.transform(coordinates[:, 2])
        new_coordinates[:, 5] = self.z_scaler.transform(coordinates[:, 5])
        
        return new_coordinates
    
    def inverse_transform(self, coordinates):
        
        new_coordinates = np.copy(coordinates)
        new_coordinates[:, 0] = self.x_scaler.inverse_transform(coordinates[:, 0])
        new_coordinates[:, 3] = self.x_scaler.inverse_transform(coordinates[:, 3])
        new_coordinates[:, 1] = self.y_scaler.inverse_transform(coordinates[:, 1])
        new_coordinates[:, 4] = self.y_scaler.inverse_transform(coordinates[:, 4])
        new_coordinates[:, 2] = self.z_scaler.inverse_transform(coordinates[:, 2])
        new_coordinates[:, 5] = self.z_scaler.inverse_transform(coordinates[:, 5])
        
        return new_coordinates
    
    def parameters_inverse_transform(self, new_params):

        return new_params * np.array([self.x_coeff, self.x_coeff / self.z_coeff, self.y_coeff, self.y_coeff / self.z_coeff]).T
    
    def parameters_transform(self, params):

        return params * np.array([1. / self.x_coeff, self.z_coeff / self.x_coeff, 1. / self.y_coeff, self.z_coeff / self.y_coeff]).T
        
######################################################################################################################################

class RetinaTrackReconstruction(object):
    
    def __init__(self):

        self.labels_ = None
        self.tracks_params_ = None
        
    def grad_R(self, a):
        
        grad_R = 0
    
        for i in range(len(self.tubes_z0s)):

            rho = z_distance(a, self.tubes_z0s[i], self.tubes_starts[i], self.tubes_directions[i])
            grad_rho = z_distance_grad(a, self.tubes_z0s[i], self.tubes_starts[i], self.tubes_directions[i])
            grad_R += np.exp(-rho**2/self.sigma**2)*rho*grad_rho

        grad_R = -1./self.sigma**2 * grad_R
        
        return -grad_R
    
    def R(self, a):
        
        R = 0
    
        for i in range(len(self.tubes_z0s)):

            R += np.exp(-z_distance(a, self.tubes_z0s[i], self.tubes_starts[i], self.tubes_directions[i])**2/self.sigma**2)
        
        return -R
    
    def minimize(self, a):
    
        l_min = minimize_scalar(lambda l: self.R(a - l * self.grad_R(a))).x
        return a - l_min * self.grad_R(a)
    
    def grad_step(self, a):
        
        return self.minimize(a)
    
    def gradient_descent(self, initial_dot):
        
        dots = [initial_dot]
        
        while self.sigma>0.001:
            
            dots.append(self.grad_step(dots[-1]))
            
            while np.linalg.norm(dots[-2]-dots[-1])>self.eps:
                dots.append(self.grad_step(dots[-1]))
            
            self.sigma = self.sigma * 0.5
            
        dots = np.array(dots)
            
        return dots

    def fit(self, ends_of_strawtubes, initial_dots):
        
        scaler = Scaler()
        normed_ends = scaler.transform(ends_of_strawtubes)
        
        starts = []
        directions = []
        z0s = []
        
        for i in range(len(normed_ends)):
            
            start, direction, z0 = ends2params(normed_ends[i])
            starts.append(start)
            directions.append(direction)
            z0s.append(z0)
        
        self.tubes_starts = np.array(starts)
        self.tubes_directions = np.array(directions)
        self.tubes_z0s = np.array(z0s)
        
        initial_sigma = 5
        self.eps = 0.000005
   
        dots = []
        
        for idot in initial_dots:
            
            self.sigma = initial_sigma
            idot = scaler.parameters_transform(idot)

            dots.append(scaler.parameters_inverse_transform(self.gradient_descent(idot)))

        return np.array(dots)