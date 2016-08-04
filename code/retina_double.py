__author__ = "oalenkin"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from scipy.optimize import minimize_scalar


######################################################################################################################################

param_vec = T.vector("param_vec", dtype='float64')
matrix_conv = T.matrix("matrix_conv", dtype='float64')
tube_vec0 = T.vector("tube_vec0", dtype='float64')
tube_vec1 = T.vector("tube_vec1", dtype='float64')

point_in_z = T.dot(matrix_conv, param_vec)
z_dist = T.sqrt(T.sum((-T.sum((tube_vec0-point_in_z)*tube_vec1)*tube_vec1+(tube_vec0-point_in_z))**2))
z_distance_theano = theano.function([param_vec, matrix_conv, tube_vec0, tube_vec1], z_dist)
z_distance_theano_grad = theano.function([param_vec, matrix_conv, tube_vec0, tube_vec1], T.grad(z_dist, param_vec))

def z_distance(param_vec, z, tube_vec0, tube_vec1):
    
    matrix_conv = np.array([[1,z,0,0],[0,0,1,z]])
    
    return z_distance_theano(param_vec, matrix_conv, tube_vec0, tube_vec1)

def z_distance_grad(param_vec, z, tube_vec0, tube_vec1):
    
    matrix_conv = np.array([[1,z,0,0],[0,0,1,z]])
    
    return z_distance_theano_grad(param_vec, matrix_conv, tube_vec0, tube_vec1)

######################################################################################################################################

param_vect = T.vector("param_vect", dtype='float64')
matrix_convert = T.tensor3("matrix_convert", dtype='float64')
tube_vec0s = T.matrix("tube_vec0s", dtype='float64')
tube_vec1s = T.matrix("tube_vec1s", dtype='float64')
sig = T.scalar("sig", dtype="float64")
dist2Wire = T.vector("dist2Wire", dtype="float64")

rs, updates = theano.scan(fn = lambda tube0, tube1, matrix, d, param_v, s:
                          T.exp(-(T.sqrt(T.sum((-T.dot(tube0-T.dot(matrix, param_v), tube1)*tube1+(tube0-T.dot(matrix, param_v)))**2))-d)**2/s**2),
                          sequences=[tube_vec0s, tube_vec1s, matrix_convert, dist2Wire],
                          non_sequences=[param_vect, sig])
    
r = rs.sum()
R_func = theano.function([param_vect, tube_vec0s, tube_vec1s, sig, matrix_convert, dist2Wire], r)

derivative_r = T.grad(r, param_vect)
derivative_R = theano.function([param_vect, tube_vec0s, tube_vec1s, sig, matrix_convert, dist2Wire], derivative_r)

def artificial_retina_response(track_params, ends_of_strawtubes, sigma, dists):
    
    starts = []
    directions = []
    matrixes = []
    
    for i in range(len(ends_of_strawtubes)):
        
        start, direction, z0 = ends2params(ends_of_strawtubes[i])
        starts.append(start)
        directions.append(direction)
        matrixes.append(np.array([[1,z0,0,0],[0,0,1,z0]]))
    
    starts = np.array(starts)
    directions = np.array(directions)
    matrixes = np.array(matrixes)
    
    return R_func(track_params, starts, directions, sigma, matrixes, dists)

######################################################################################################################################

def ends2params(array):
    
    start = array[:2]
    k = (array[4]-array[1])/(array[3]-array[0])
    direction = np.array([1., k])/np.sqrt(1+k**2)
    z0 = array[5]
    
    return start, direction, z0

######################################################################################################################################

def artificial_retina_response_grad(track_params, ends_of_strawtubes, sigma, dists):
    
    starts = []
    directions = []
    matrixes = []
    
    for i in range(len(ends_of_strawtubes)):
        
        start, direction, z0 = ends2params(ends_of_strawtubes[i])
        starts.append(start)
        directions.append(direction)
        matrixes.append(np.array([[1,z0,0,0],[0,0,1,z0]]))
    
    starts = np.array(starts)
    directions = np.array(directions)
    matrixes = np.array(matrixes)
    
    return derivative_R(track_params, starts, directions, sigma, matrixes, dists)

def artificial_retina_response_grad2(track_params, ends_of_strawtubes, sigma, dists):
    
    grad_R = 0
    
    for i in range(len(ends_of_strawtubes)):
    
        start, direction, z0 = ends2params(ends_of_strawtubes[i])
        rho = np.abs(z_distance(track_params, z0, start, direction)-dists[i])
        grad_rho = z_distance_grad(track_params, z0, start, direction)
        grad_R += np.exp(-rho**2/sigma**2)*rho*grad_rho
        
        print i, dists[i], rho, grad_rho, start, direction, z0
    
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

def plot_artificial_retina_response(event, params_array, sigma, log=False):
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
    
    if log == True:
        projection_on_yz = np.log10(projection_on_yz)
        projection_on_xz = np.log10(projection_on_xz)
    
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
    _ = plt.imshow(projection_on_xz, aspect='auto', cmap=plt.cm.Oranges, extent=(ls.min(), ls.max(), x0s.max(), x0s.min()), interpolation='None')
    #_ = plt.contourf(x0s, ls, projection_on_xz, cmap=plt.cm.Oranges)
    plt.xlabel("x0")
    plt.ylabel("l")
    plt.colorbar()
    
    fig2 = plt.figure(figsize=(9, 7))
    plt.title("Projection on YZ", fontsize=20, fontweight='bold')
    _ = plt.imshow(projection_on_yz, aspect='auto', cmap=plt.cm.Oranges, extent=(ms.min(), ms.max(), y0s.max(), y0s.min()), interpolation='None')
    #_ = plt.contourf(y0s, ms, projection_on_yz, cmap=plt.cm.Oranges)
    plt.xlabel("y0")
    plt.ylabel("m")
    plt.colorbar()
            
    return fig1, fig2

######################################################################################################################################

class Scaler():
    
    def __init__(self, z_scale=1., y_scale=1., x_scale=1.):
        
        self.z_scale = z_scale
        self.y_scale = y_scale
        self.x_scale = x_scale
        
    def transform(self, coordinates):
        
        new_coordinates = np.copy(coordinates)
        new_coordinates[:, 0] = coordinates[:, 0] / self.x_scale
        new_coordinates[:, 3] = coordinates[:, 3] / self.x_scale
        new_coordinates[:, 1] = coordinates[:, 1] / self.y_scale
        new_coordinates[:, 4] = coordinates[:, 4] / self.y_scale
        new_coordinates[:, 2] = coordinates[:, 2] / self.z_scale
        new_coordinates[:, 5] = coordinates[:, 5] / self.z_scale
        
        return new_coordinates
    
    def inverse_transform(self, coordinates):
        
        new_coordinates = np.copy(coordinates)
        new_coordinates[:, 0] = coordinates[:, 0] * self.x_scale
        new_coordinates[:, 3] = coordinates[:, 3] * self.x_scale
        new_coordinates[:, 1] = coordinates[:, 1] * self.y_scale
        new_coordinates[:, 4] = coordinates[:, 4] * self.y_scale
        new_coordinates[:, 2] = coordinates[:, 2] * self.z_scale
        new_coordinates[:, 5] = coordinates[:, 5] * self.z_scale
        
        return new_coordinates
    
    def parameters_inverse_transform(self, new_params):

        return new_params * np.array([self.x_scale, self.x_scale / self.z_scale, self.y_scale, self.y_scale / self.z_scale]).T
    
    def parameters_transform(self, params):

        return params * np.array([1. / self.x_scale, self.z_scale / self.x_scale, 1. / self.y_scale, self.z_scale / self.y_scale]).T
    
######################################################################################################################################

def distances(track_params, tubes_starts, tubes_directions, tubes_zs, dists):
    
    distances = []
    
    for i in range(len(tubes_zs)):
        
        distances.append(z_distance_theano(track_params, np.array([[1, tubes_zs[i], 0, 0],[0, 0, 1, tubes_zs[i]]]), tubes_starts[i], tubes_directions[i]))
            
    distances = np.array(distances)
        
    return distances - dists
    
######################################################################################################################################

class RetinaTrackReconstruction(object):
    
    def __init__(self, z_scale=1500., sigma_from=0.8, sigma_to=0.8, y_scale=1., x_scale=1., stopping_criteria=0.00001, \
                sigma_decrement=0.5, noise_treshold=2., inlier_treshold=0.8, pre_sigma=0.8):

        self.labels_ = None
        self.tracks_params_ = None
        self.z_scale = z_scale
        self.y_scale = y_scale
        self.x_scale = x_scale
        self.sigma_from = sigma_from
        self.sigma_to = sigma_to
        self.stopping_criteria = stopping_criteria
        self.sigma_decrement = sigma_decrement
        self.noise_treshold = noise_treshold
        self.inlier_treshold = inlier_treshold
        self.pre_sigma = pre_sigma
        
    def grad_R(self, a, tubes_starts, tubes_directions, sigma, matrixes, dists):
        
        return -derivative_R(a, tubes_starts, tubes_directions, sigma, matrixes, dists)
    
    def R(self, a, tubes_starts, tubes_directions, sigma, matrixes, dists):
        
        return -R_func(a, tubes_starts, tubes_directions, sigma, matrixes, dists)
    
    def step_on_direction(self, a, direction, tubes_starts, tubes_directions, sigma, matrixes, dists):
        
        l_min = minimize_scalar(lambda l: self.R(a - l * direction, tubes_starts, tubes_directions, sigma, matrixes, dists), tol=0.1, method='brent').x
                                
        return a - l_min * direction
    
    def conjugate_gradient_method(self, initial_dot, tubes_starts, tubes_directions, matrixes, dists):

        dots = [initial_dot]
        
        sigma = self.sigma_from
        
        while sigma >= self.sigma_to:
            
            r = [0, 0]
            r[1] = self.grad_R(dots[-1], tubes_starts, tubes_directions, sigma, matrixes, dists)
            beta = 0
            direction = r[1]
            dots.append(self.step_on_direction(dots[-1], direction, tubes_starts, tubes_directions, sigma, matrixes, dists))
            last_direction = direction
            counter = 1
            
            while np.linalg.norm(dots[-1]-dots[-2])>self.stopping_criteria/self.sigma_to*sigma:
            
                while counter < 4:

                    r[0] = r[1]
                    r[1] = self.grad_R(dots[-1], tubes_starts, tubes_directions, sigma, matrixes, dists)
                    beta = np.dot(r[1], r[1]-r[0]) / np.dot(r[0], r[0])
                    direction = r[1] + beta * last_direction
                    dots.append(self.step_on_direction(dots[-1], direction, tubes_starts, tubes_directions, sigma, matrixes, dists))
                    last_direction = direction

                    counter += 1

                r[0] = r[1]
                r[1] = self.grad_R(dots[-1], tubes_starts, tubes_directions, sigma, matrixes, dists)
                beta = np.max(beta, 0)
                direction = r[1] + beta * last_direction
                dots.append(self.step_on_direction(dots[-1], direction, tubes_starts, tubes_directions, sigma, matrixes, dists))
                last_direction = direction
                counter = 1
                
            sigma *= self.sigma_decrement
                
        return dots
    
    def initial_dots_generator(self, ends):
        
        #constants
        delta = 15
        
        xs = []
        ys = []
        zs = []
        
        for i in range(len(ends)-1):
            
            for j in range(i+1, len(ends)):
                
                one = ends[i]
                two = ends[j]
                
                if np.abs(one[2]-two[2])<delta and (one[1]-two[1])*(one[4]-two[4])<0:
                    
                    y1 = one[1]
                    y2 = one[4]
                    x1 = one[0]
                    x2 = one[3]
                    z12 = one[2]
                    k12 = (y2 - y1) / (x2 - x1)
                    b12 = y1 - k12 * x1
                    y3 = two[1]
                    y4 = two[4]
                    x3 = two[0]
                    x4 = two[3]
                    z34 = two[2]
                    k34 = (y4 - y3) / (x4 - x3)
                    b34 = y3 - k34 * x3
                    x = (b12 - b34) / (k34 - k12)
                    y = k12 * x + b12
                    z = 0.5 * (z12 + z34)

                    if np.abs(x) <= 300:

                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        
        init_dots = []
                        
        for i in range(len(zs)):
            
            for j in range(i+1, len(zs)):
                
                if zs[i] - zs[j] > delta:
                    
                    x1 = xs[i]
                    x2 = xs[j]
                    y1 = ys[i]
                    y2 = ys[j]
                    z1 = zs[i]
                    z2 = zs[j]
                    k_y = (y2 - y1) / (z2 - z1)
                    b_y = y1 - k_y * z1
                    k_stereo = (x2 - x1) / (z2 - z1)
                    b_stereo = x1 - k_stereo * z1
                
                    init_dots.append([b_stereo, k_stereo, b_y+0.000001, k_y])
        
        return np.array(init_dots)
         
    def fit(self, ends_of_strawtubes, dists):
        
        scaler = Scaler(z_scale=self.z_scale, y_scale=self.y_scale, x_scale=self.x_scale)
        normed_ends = scaler.transform(ends_of_strawtubes)
        
        deltas = []
        for d in dists:
            deltas.append(np.array([0, d]))
        dists = np.zeros(len(dists))
        deltas = np.array(deltas)
        
        starts = []
        directions = []
        z0s = []
        matrixes = []
        
        for i in range(len(normed_ends)):
            
            start, direction, z0 = ends2params(normed_ends[i])
            starts.append(start)
            directions.append(direction)
            z0s.append(z0)
            matrixes.append(np.array([[1,z0,0,0],[0,0,1,z0]]))
        
        tubes_starts = np.array(starts)
        tubes_directions = np.array(directions)
        tubes_z0s = np.array(z0s)
        matrixes = np.array(matrixes)

        initial_dots = self.initial_dots_generator(ends_of_strawtubes)
   
        min_R = 0
        best_dot = np.array([0,0,0,0])
        
        for idot in initial_dots:
            
            idot_tr = scaler.parameters_transform(idot)

            new_R = self.R(idot_tr, tubes_starts, tubes_directions, self.pre_sigma, matrixes, dists)
            
            if new_R < min_R:
                
                min_R = new_R
                best_dot = idot_tr
        
        #return scaler.parameters_inverse_transform(self.conjugate_gradient_method(best_dot, tubes_starts,\
        #                                                                          tubes_directions, matrixes))
        
        t_s = np.array(list(tubes_starts+deltas)+list(tubes_starts-deltas))
        t_d = np.array(list(tubes_directions)*2)
        mats = np.array(list(matrixes)*2)
        dst = np.zeros(len(t_s))

        track1 = self.conjugate_gradient_method(best_dot, t_s, t_d, mats, dst)[-1]
        distances1 = distances(track1, tubes_starts, tubes_directions, tubes_z0s, dists)
        mask = distances1 > self.inlier_treshold
        
        labels = np.array([-1] * len(distances1))
            
        initial_dots = self.initial_dots_generator(ends_of_strawtubes[mask])
        
        t_s = np.array(list(tubes_starts[mask]+deltas[mask])+list(tubes_starts[mask]-deltas[mask]))
        t_d = np.array(list(tubes_directions[mask])*2)
        mats = np.array(list(matrixes[mask])*2)
        dst = np.zeros(len(t_s))
        
        if len(initial_dots)>0:

            min_R = 0
            best_dot = np.array([0,0,0,0])

            for idot in initial_dots:
            
                idot_tr = scaler.parameters_transform(idot)

                new_R = self.R(idot_tr, tubes_starts[mask], tubes_directions[mask], self.pre_sigma, matrixes[mask], dists[mask])

                if (new_R <= min_R):

                    min_R = new_R
                    best_dot = idot_tr
                    
            track2 = self.conjugate_gradient_method(best_dot, t_s, t_d, mats, dst)[-1]
        
        else:
            
            buf = self.sigma_from
            self.sigma_from = 200
            track2 = self.conjugate_gradient_method(np.array([0, 0, 0, 0]), t_s, t_d, mats, dst)[-1]
            self.sigma_from = buf
                
        distances2 = distances(track2, tubes_starts, tubes_directions, tubes_z0s, dists)

        for i in range(len(distances1)):

            if distances1[i] < np.min([distances2[i], self.noise_treshold]):

                labels[i] = 0

            elif distances2[i] < np.min([distances1[i], self.noise_treshold]):

                labels[i] = 1
                
        self.labels_ = labels
        
        track1 = scaler.parameters_inverse_transform(track1)
        track2 = scaler.parameters_inverse_transform(track2)
        
        self.tracks_params_ = np.array([[[track1[3], track1[2]], [track1[1], track1[0]]], [[track2[3], track2[2]],[track2[1], track2[0]]]])
        #self.tracks_params_ = np.array([track1, track2])