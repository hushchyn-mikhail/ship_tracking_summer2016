__author__ = "oalenkin"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from scipy.optimize import minimize_scalar
from skimage.draw import line_aa


######################################################################################################################################

z_magnet = T.scalar("z_magnet", dtype='float64')

m = T.scalar("m", dtype='float64')
l = T.scalar('l_before', dtype='float64')
x0 = T.scalar('x0', dtype='float64')
y0 = T.scalar('y0', dtype='float64')
z_tube = T.scalar('z_tube', dtype='float64')
tube_vec0 = T.vector('tube_vec0', dtype='float64')
tube_vec1 = T.vector('tube_vec1', dtype='float64')

point_in_z = [x0 + (z_tube - z_magnet) * m, y0 + (z_tube - z_magnet) * l]
z_dist = T.sqrt(T.sum((-T.sum((tube_vec0 - point_in_z) * tube_vec1) * tube_vec1 + \
                              (tube_vec0 - point_in_z)) ** 2))
z_distance_theano = theano.function([x0, m, y0, l, z_tube, tube_vec0, tube_vec1, z_magnet], z_dist)

def z_distance(x0, m, y0, l, z_tube, tube_vec0, tube_vec1, z_magnet=3070.):
    
    return z_distance_theano(x0, m, y0, l, z_tube, tube_vec0, tube_vec1, z_magnet)
    
######################################################################################################################################

l_b = T.scalar('l_b', dtype='float64')
l_a = T.scalar('l_a', dtype='float64')

tube_zs_b = T.vector('tube_zs_b', dtype='float64')
tube_vec0s_b = T.matrix("tube_vec0s_b", dtype='float64')
tube_vec1s_b = T.matrix("tube_vec1s_b", dtype='float64')
dist2Wires_b = T.vector("dist2Wire_b", dtype="float64")

tube_zs_a = T.vector('tube_zs_a', dtype='float64')
tube_vec0s_a = T.matrix("tube_vec0s_a", dtype='float64')
tube_vec1s_a = T.matrix("tube_vec1s_a", dtype='float64')
dist2Wires_a = T.vector("dist2Wire_a", dtype="float64")

sig = T.scalar("sig", dtype="float64")

rs_b, updates = theano.scan(fn = lambda tube0, tube1, z, d, x_t, m_t, y_t, l_t, s, z_m:
                          T.exp(-(T.sqrt(T.sum((-T.dot(tube0 - [x_t + (z - z_m) * m_t, y_t + (z - z_m) * l_t], tube1) *\
                                                tube1 + (tube0 - [x_t + (z - z_m) * m_t, y_t + (z - z_m) * l_t])) ** 2)) -\
                                  d) ** 2/s ** 2),
                          sequences=[tube_vec0s_b, tube_vec1s_b, tube_zs_b, dist2Wires_b],
                          non_sequences=[x0, m, y0, l_b, sig, z_magnet])

rs_a, updates = theano.scan(fn = lambda tube0, tube1, z, d, x_t, m_t, y_t, l_t, s, z_m:
                          T.exp(-(T.sqrt(T.sum((-T.dot(tube0 - [x_t + (z - z_m) * m_t, y_t + (z - z_m) * l_t], tube1) *\
                                                tube1 + (tube0 - [x_t + (z - z_m) * m_t, y_t + (z - z_m) * l_t])) ** 2)) -\
                                  d) ** 2/s ** 2),
                          sequences=[tube_vec0s_a, tube_vec1s_a, tube_zs_a, dist2Wires_a],
                          non_sequences=[x0, m, y0, l_a, sig, z_magnet])
    
r_a = rs_a.sum()
r_b = rs_b.sum()
r = r_a + r_b
R_func = theano.function([x0, m, y0, l_b, l_a, tube_vec0s_b, tube_vec1s_b, tube_zs_b, dist2Wires_b, tube_vec0s_a, tube_vec1s_a,\
                          tube_zs_a, dist2Wires_a, sig, z_magnet], r)

derivative_r = T.grad(r, [x0, m, l_b, l_a])
derivative_R = theano.function([x0, m, y0, l_b, l_a, tube_vec0s_b, tube_vec1s_b, tube_zs_b, dist2Wires_b, tube_vec0s_a, tube_vec1s_a,\
                                tube_zs_a, dist2Wires_a, sig, z_magnet], derivative_r)

def artificial_retina_response(params, y, ends_of_strawtubes_b, dists_b, ends_of_strawtubes_a, dists_a, sigma, z_magnet=3070.):
    
    x, m, l_b, l_a = params
    
    starts_b = []
    directions_b = []
    z0s_b = []
    
    for i in range(len(ends_of_strawtubes_b)):
        
        start, direction, z0 = ends2params(ends_of_strawtubes_b[i])
        starts_b.append(start)
        directions_b.append(direction)
        z0s_b.append(z0)
    
    starts_b = np.array(starts_b)
    directions_b = np.array(directions_b)
    z0s_b = np.array(z0s_b)
    
    starts_a = []
    directions_a = []
    z0s_a = []
    
    for i in range(len(ends_of_strawtubes_a)):
        
        start, direction, z0 = ends2params(ends_of_strawtubes_a[i])
        starts_a.append(start)
        directions_a.append(direction)
        z0s_a.append(z0)
    
    starts_a = np.array(starts_a)
    directions_a = np.array(directions_a)
    z0s_a = np.array(z0s_a)
    
    return R_func(x, m, y, l_b, l_a, starts_b, directions_b, z0s_b, dists_b, starts_a, directions_a, z0s_a, dists_a, sigma, z_magnet)

######################################################################################################################################

def ends2params(array):
    
    start = array[:2]
    k = (array[4]-array[1])/(array[3]-array[0])
    direction = np.array([1., k])/np.sqrt(1+k**2)
    z0 = array[5]
    
    return start, direction, z0

######################################################################################################################################

def artificial_retina_response_grad(params, y, ends_of_strawtubes_b, dists_b, ends_of_strawtubes_a, dists_a, sigma, z_magnet=3070.):
    
    x, m, l_b, l_a = params
    
    starts_b = []
    directions_b = []
    z0s_b = []
    
    for i in range(len(ends_of_strawtubes_b)):
        
        start, direction, z0 = ends2params(ends_of_strawtubes_b[i])
        starts_b.append(start)
        directions_b.append(direction)
        z0s_b.append(z0)
    
    starts_b = np.array(starts_b)
    directions_b = np.array(directions_b)
    z0s_b = np.array(z0s_b)
    
    starts_a = []
    directions_a = []
    z0s_a = []
    
    for i in range(len(ends_of_strawtubes_a)):
        
        start, direction, z0 = ends2params(ends_of_strawtubes_a[i])
        starts_a.append(start)
        directions_a.append(direction)
        z0s_a.append(z0)
    
    starts_a = np.array(starts_a)
    directions_a = np.array(directions_a)
    z0s_a = np.array(z0s_a)
    
    return derivative_R(x, m, y, l_b, l_a, starts_b, directions_b, z0s_b, dists_b, starts_a, directions_a, z0s_a, dists_a, sigma,\
                        z_magnet)

######################################################################################################################################

def get_track_params(event, trackID, z_magnet=3070.):
    """
    Returns x0, l, y0, m parameters of track.
    Input:
        event - pandas dataframe containing all hits of any event before/after magnet;
        trackID - id of track.
    Output:
        [x0, l, y0, m] - list of parametres of track.
    """
    track = event[event.TrackID==trackID]
    
    track_before = track[track.StatNb < 3]
    track_after = track[track.StatNb > 2]
    
    Xs = track_before.X.values
    Ys = track_before.Y.values
    Zs = track_before.Z.values
    
    x_params = np.polyfit(Zs, Xs, 1)
    l = x_params[0]
    x0 = x_params[1] + l * z_magnet

    y_params = np.polyfit(Zs, Ys, 1)
    m_before = y_params[0]
    y0 = y_params[1] + m_before * z_magnet
    
    Ys = track_after.Y.values
    Zs = track_after.Z.values

    y_params = np.polyfit(Zs, Ys, 1)
    m_after = y_params[0]
    
    return [x0, l, m_before, m_after], y0

######################################################################################################################################

class Scaler():
    
    def __init__(self, z_scale=1., y_scale=1., x_scale=1.):
        """
        This class implements linear transformation of space.
        :z_scale: coefficiet of compression along z-axis.
        :y_scale: coefficiet of compression along y-axis.
        :x_scale: coefficiet of compression along x-axis.
        :return:
        """
        
        self.z_scale = z_scale
        self.y_scale = y_scale
        self.x_scale = x_scale
        
    def transform(self, coordinates):
        """
        Transfroms numpy-matrix of coordinates according to compression coefficiets.
        :coordinates: numpy-matrix columns of that are [Wx1, Wy1, Wz, Wx2, Wy2, Wz].
        :return: transformed coordinates in the same format.
        """
        
        new_coordinates = np.copy(coordinates)
        new_coordinates[:, 0] = coordinates[:, 0] / self.x_scale
        new_coordinates[:, 3] = coordinates[:, 3] / self.x_scale
        new_coordinates[:, 1] = coordinates[:, 1] / self.y_scale
        new_coordinates[:, 4] = coordinates[:, 4] / self.y_scale
        new_coordinates[:, 2] = coordinates[:, 2] / self.z_scale
        new_coordinates[:, 5] = coordinates[:, 5] / self.z_scale
        
        return new_coordinates
    
    def inverse_transform(self, coordinates):
        """
        This method implements inverse transformation.
        :coordinates: numpy-matrix columns of that are [Wx1, Wy1, Wz, Wx2, Wy2, Wz].
        :return: transformed coordinates in the same format.
        """
        
        new_coordinates = np.copy(coordinates)
        new_coordinates[:, 0] = coordinates[:, 0] * self.x_scale
        new_coordinates[:, 3] = coordinates[:, 3] * self.x_scale
        new_coordinates[:, 1] = coordinates[:, 1] * self.y_scale
        new_coordinates[:, 4] = coordinates[:, 4] * self.y_scale
        new_coordinates[:, 2] = coordinates[:, 2] * self.z_scale
        new_coordinates[:, 5] = coordinates[:, 5] * self.z_scale
        
        return new_coordinates
    
    def parameters_inverse_transform(self, new_params):
        """
        This method implements inverse transformation of parameters.
        :new_params: 4-vector array([x0, l, y0, m]) of parameters in transformed space.
        :return: 4-vector of parameters in original space.
        """

        return new_params * np.array([self.x_scale, self.x_scale / self.z_scale, self.y_scale / self.z_scale,\
                                      self.y_scale / self.z_scale]).T
    
    def parameters_transform(self, params):
        """
        This method implements transformation of parameters from original space to new.
        :params: 4-vector array([x0, l, y0, m]) of parameters in original space.
        :return: 4-vector of parameters in transformed space.
        """

        return params * np.array([1. / self.x_scale, self.z_scale / self.x_scale, self.z_scale / self.y_scale,\
                                  self.z_scale / self.y_scale]).T
    
######################################################################################################################################

def projection_on_magnet(ends_of_strawtubes, dists, x_resolution=1., y_resolution=1., z_magnet=3070.):
    
    tubes = []
    for i in range(len(ends_of_strawtubes)):
        tubes.append(ends2params(ends_of_strawtubes[i]))

    lines = []
    z3 = z_magnet
    for i in range(len(tubes)):
        for j in range(i+1, len(tubes)):
            t1 = tubes[i]
            t2 = tubes[j]
            d1 = dists[i]
            d2 = dists[j]
            shift1 = np.array([0, d1])
            shift2 = np.array([0, d2])
            if (t1[2] != t2[2]) and (np.sum(t1[1] - t2[1]) < 0.01):
                start1 = t1[0]
                start2 = t2[0]
                z1 = t1[2]
                z2 = t2[2]
                start3 = ((start2 + shift2) - (start1 + shift1)) / (z2 - z1) * (z3 - z2) + start2
                if (start3[1] > -500) and (start3[1] < 500) and (start3[0] < -225) and (start3[0] > -275):
                    lines.append([start3, t1[1]])

                start3 = ((start2 - shift2) - (start1 + shift1)) / (z2 - z1) * (z3 - z2) + start2
                if (start3[1] > -500) and (start3[1] < 500) and (start3[0] < -225) and (start3[0] > -275):
                    lines.append([start3, t1[1]])

                start3 = ((start2 + shift2) - (start1 - shift1)) / (z2 - z1) * (z3 - z2) + start2
                if (start3[1] > -500) and (start3[1] < 500) and (start3[0] < -225) and (start3[0] > -275):
                    lines.append([start3, t1[1]])

                start3 = ((start2 - shift2) - (start1 - shift1)) / (z2 - z1) * (z3 - z2) + start2
                if (start3[1] > -500) and (start3[1] < 500) and (start3[0] < -225) and (start3[0] > -275):
                    lines.append([start3, t1[1]])
    
    x_len = int(600 * x_resolution)
    y_len = int(1200 * y_resolution)
    line_len = 500

    matrix = np.zeros([x_len, y_len])
    for line in lines:
        rr, cc, val = line_aa(int(line[0][0] * x_resolution) + x_len / 2, int(line[0][1] * y_resolution) + y_len / 2,\
                              int((line[0][0] + line[1][0] * line_len) * x_resolution) + x_len / 2,\
                              int((line[0][1] + line[1][1] * line_len) * y_resolution) + y_len / 2)
        matrix[rr, cc] += 1
        
    return matrix
    
######################################################################################################################################

class RetinaTrackReconstruction(object):
    
    def __init__(self, z_scale=1500., sigma_from=0.8, sigma_to=0.8, y_scale=1., x_scale=1., stopping_criteria=0.00001,\
                 z_magnet=3070., sigma_decrement=0.5, noise_treshold=2., inlier_treshold=0.8, pre_sigma=0.8, x_resolution=1.,\
                 y_resolution=1.):
        """
        This class recognizes 2 tracks before/after magnet.
        :z_scale: coefficient of compression of original space(X,Y,Z) along z_axis.
        :y_scale: coefficient of compression of original space(X,Y,Z) along y_axis.
        :x_scale: coefficient of compression of original space(X,Y,Z) along x_axis.
        :sigma_from: starting value of sigma in process of optimization.
        :sigma_to: finishing value of sigma in process of optimization.
        :sigma_decrement: speed of decreasing.
        :pre_sigma: value of sigma for selection of initial tracks.
        :noise_treshold: treshold to label hit as noise.
        :inlier_treshold: treshold to label hit as inlier.
        :return:
        """

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
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.z_magnet = z_magnet
        
    def grad_R(self, params, y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
               tubes_directions_a, tubes_zs_a, dists_a, sigma):
        
        x0, m, l_b, l_a = params
        
        return -np.array(derivative_R(x0, m, y0, l_b, l_a, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                             tubes_directions_a, tubes_zs_a, dists_a, sigma, self.z_magnet))
    
    def R(self, params, y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a, tubes_directions_a,\
          tubes_zs_a, dists_a, sigma):
        
        x0, m, l_b, l_a = params
        
        return -R_func(x0, m, y0, l_b, l_a, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                       tubes_directions_a, tubes_zs_a, dists_a, sigma, self.z_magnet)
    
    def step_on_direction(self, params, direction, y0, tubes_starts_b, tubes_directions_b, tubes_zs_b,\
                          dists_b, tubes_starts_a, tubes_directions_a, tubes_zs_a, dists_a, sigma):
        
        l_min = minimize_scalar(lambda l: self.R(params - l * direction, y0, tubes_starts_b,\
                                                 tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a, tubes_directions_a,\
                                                 tubes_zs_a, dists_a, sigma),\
                                tol=0.1, method='brent').x
                                
        return params - l_min * direction
    
    def conjugate_gradient_method(self, params, y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                                  tubes_directions_a, tubes_zs_a, dists_a):
        """
        This method implements process of optimization.
        :initial_dot: starting dot for optimization.
        :tubes_starts: array of any points of tubes.
        :tubes_directions: array of directing vectors of tubes.
        :matrixes: matrixes containing z-coordinate.
        :dists: array of dist2Wire.
        :return: optimal dot.
        """

        dots = [params]
        
        sigma = self.sigma_from
        #####################
        values = [self.R(dots[-1], y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                              tubes_directions_a, tubes_zs_a, dists_a, sigma)]
        
        while sigma >= self.sigma_to:
            
            r = [0, 0]
            r[1] = self.grad_R(dots[-1], y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                               tubes_directions_a, tubes_zs_a, dists_a, sigma)
            beta = 0
            direction = r[1]
            dots.append(self.step_on_direction(dots[-1], direction, y0, tubes_starts_b, tubes_directions_b, tubes_zs_b,\
                                               dists_b, tubes_starts_a, tubes_directions_a, tubes_zs_a, dists_a, sigma))
            ##################
            values.append(self.R(dots[-1], y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                              tubes_directions_a, tubes_zs_a, dists_a, sigma))
            last_direction = direction
            counter = 1
            
            while np.linalg.norm(dots[-1]-dots[-2])>self.stopping_criteria/self.sigma_to*sigma:
            
                while counter < 4:

                    r[0] = r[1]
                    r[1] = self.grad_R(dots[-1], y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                                       tubes_directions_a, tubes_zs_a, dists_a, sigma)
                    beta = np.dot(r[1], r[1]-r[0]) / np.dot(r[0], r[0])
                    direction = r[1] + beta * last_direction
                    dots.append(self.step_on_direction(dots[-1], direction, y0, tubes_starts_b, tubes_directions_b, tubes_zs_b,\
                                                       dists_b, tubes_starts_a, tubes_directions_a, tubes_zs_a, dists_a, sigma))
                    ##################
                    values.append(self.R(dots[-1], y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                                      tubes_directions_a, tubes_zs_a, dists_a, sigma))
                    last_direction = direction

                    counter += 1

                r[0] = r[1]
                r[1] = self.grad_R(dots[-1], y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                                   tubes_directions_a, tubes_zs_a, dists_a, sigma)
                beta = np.max(beta, 0)
                direction = r[1] + beta * last_direction
                dots.append(self.step_on_direction(dots[-1], direction, y0, tubes_starts_b,\
                                                   tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                                                   tubes_directions_a, tubes_zs_a, dists_a, sigma))
                ##################
                values.append(self.R(dots[-1], y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a,\
                                  tubes_directions_a, tubes_zs_a, dists_a, sigma))
                last_direction = direction
                counter = 1
                
            sigma *= self.sigma_decrement
        ################        
        return dots, values
    
    def candidates_generator(self, ends_b, ends_a, y0):
        """
        This method generates initial tracks.
        :ends: numpy-matrix of coordinates of hits [Wx1, Wy1, Wz, Wx2, Wy2, Wz].
        :return:
        """
        
        #constants
        delta = 15
        
        xs_b = []
        ys_b = []
        zs_b = []
        
        for i in range(len(ends_b)-1):
            
            for j in range(i+1, len(ends_b)):
                
                one = ends_b[i]
                two = ends_b[j]
                
                if np.abs(one[2]-two[2])<delta and (one[1]-two[1])*(one[4]-two[4])<0:
                    
                    x1, y1, z12, x2, y2, extra = one
                    k12 = (y2 - y1) / (x2 - x1)
                    b12 = y1 - k12 * x1
                    x3, y3, z34, x4, y4, extra = two
                    k34 = (y4 - y3) / (x4 - x3)
                    b34 = y3 - k34 * x3
                    x = (b12 - b34) / (k34 - k12)
                    y = k12 * x + b12
                    z = 0.5 * (z12 + z34)

                    if np.abs(x) <= 300:

                        xs_b.append(x)
                        ys_b.append(y)
                        zs_b.append(z)
                        
        xs_a = []
        ys_a = []
        zs_a = []
        
        for i in range(len(ends_a)-1):
            
            for j in range(i+1, len(ends_a)):
                
                one = ends_a[i]
                two = ends_a[j]
                
                if np.abs(one[2]-two[2])<delta and (one[1]-two[1])*(one[4]-two[4])<0:
                    
                    x1, y1, z12, x2, y2, extra = one
                    k12 = (y2 - y1) / (x2 - x1)
                    b12 = y1 - k12 * x1
                    x3, y3, z34, x4, y4, extra = two
                    k34 = (y4 - y3) / (x4 - x3)
                    b34 = y3 - k34 * x3
                    x = (b12 - b34) / (k34 - k12)
                    y = k12 * x + b12
                    z = 0.5 * (z12 + z34)

                    if np.abs(x) <= 300:

                        xs_a.append(x)
                        ys_a.append(y)
                        zs_a.append(z)
                        
        init_dots = []
                        
        for i in range(len(zs_a)):
            
            for j in range(len(zs_b)):
                
                m = (xs_a[i] - xs_b[j]) / (zs_a[i] - zs_b[j])
                l_b = (y0 - ys_b[j]) / (self.z_magnet - zs_b[j])
                l_a = (ys_a[i] - y0) / (zs_a[i] - self.z_magnet)
                x0 = xs_b[j] + m * (self.z_magnet - zs_b[j])
                
                init_dots.append([x0, m, l_b, l_a])
        
        return np.array(init_dots)
    
    def get_y0x0(self, ends_of_strawtubes_b, dists_b, ends_of_strawtubes_a, dists_a):
        
        projection_b = projection_on_magnet(ends_of_strawtubes_b, dists_b, self.x_resolution, self.y_resolution, self.z_magnet)
        projection_a = projection_on_magnet(ends_of_strawtubes_a, dists_a, self.x_resolution, self.y_resolution, self.z_magnet)
        
        y0 = np.argmax((projection_b + projection_a).sum(axis=0))
        x0 = np.argmax((projection_b + projection_a)[:, y0])
        
        y0 = (y0 - projection_b.shape[1] * 0.5) / self.y_resolution
        x0 = (x0 - projection_b.shape[0] * 0.5) / self.x_resolution
        if abs(x0) > 450 :
            x0 = 0
        
        return y0, x0
         
    def fit(self, event):
        
        ends_of_strawtubes_b = event[event.StatNb < 3][['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz']].values
        #dists_b = event[event.StatNb < 3]['dist2Wire'].values
        dists_b = np.zeros(len(ends_of_strawtubes_b))
        ends_of_strawtubes_a = event[event.StatNb > 2][['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz']].values
        #dists_a = event[event.StatNb > 2]['dist2Wire'].values
        dists_a = np.zeros(len(ends_of_strawtubes_a))
        
        starts = []
        directions = []
        z0s = []
        
        for i in range(len(ends_of_strawtubes_b)):
            
            start, direction, z0 = ends2params(ends_of_strawtubes_b[i])
            starts.append(start)
            directions.append(direction)
            z0s.append(z0)
        
        tubes_starts_b = np.array(starts)
        tubes_directions_b = np.array(directions)
        tubes_zs_b = np.array(z0s)
        
        starts = []
        directions = []
        z0s = []
        
        for i in range(len(ends_of_strawtubes_a)):
            
            start, direction, z0 = ends2params(ends_of_strawtubes_a[i])
            starts.append(start)
            directions.append(direction)
            z0s.append(z0)
        
        tubes_starts_a = np.array(starts)
        tubes_directions_a = np.array(directions)
        tubes_zs_a = np.array(z0s)
        
        y0, x0 = self.get_y0x0(ends_of_strawtubes_b, dists_b, ends_of_strawtubes_a, dists_a)
        
        candidates = self.candidates_generator(ends_of_strawtubes_b, ends_of_strawtubes_a, y0)
        
        best_model = np.array([0, 0, 0, 0])
        min_R = 0
        
        for candidate in candidates:
        
            R = self.R(candidate, y0, tubes_starts_b, tubes_directions_b, tubes_zs_b, dists_b, tubes_starts_a, tubes_directions_a,\
                       tubes_zs_a, dists_a, self.pre_sigma)
            
            if R < min_R:
                min_R = R
                best_model = candidate
                
        x0, m, l_b, l_a = best_model
        
        mask_b = []
                
        for i in range(len(tubes_starts_b)):
            
            if z_distance(x0, m, y0, l_b, tubes_zs_b[i], tubes_starts_b[i], tubes_directions_b[i], z_magnet=self.z_magnet) < self.inlier_treshold:
                
                mask_b.append(0)
                
            else:
                
                mask_b.append(1)
                
        mask_b = np.array(mask_b, dtype=bool)
        
        mask_a = []
                
        for i in range(len(tubes_starts_a)):
            
            if z_distance(x0, m, y0, l_a, tubes_zs_a[i], tubes_starts_a[i], tubes_directions_a[i], z_magnet=self.z_magnet) < self.inlier_treshold:
                
                mask_a.append(0)
                
            else:
                
                mask_a.append(1)
                
        mask_a = np.array(mask_a, dtype=bool)
        
        y0_2, x0_2 = self.get_y0x0(ends_of_strawtubes_b[mask_b], dists_b[mask_b], ends_of_strawtubes_a[mask_a], dists_a[mask_a])
        
        best_model2 = np.array([0, 0, 0, 0])
        min_R = 0
        
        candidates = self.candidates_generator(ends_of_strawtubes_b[mask_b], ends_of_strawtubes_a[mask_a], y0_2)
        
        for candidate in candidates:
        
            R = self.R(candidate, y0_2, tubes_starts_b[mask_b], tubes_directions_b[mask_b], tubes_zs_b[mask_b], dists_b[mask_b],\
                       tubes_starts_a[mask_a], tubes_directions_a[mask_a], tubes_zs_a[mask_a], dists_a[mask_a], self.pre_sigma)
            
            if R < min_R:
                min_R = R
                best_model2 = candidate
                
        x02, m, l_b, l_a = best_model2
        
        mask_b2 = []
                
        for i in range(len(tubes_starts_b)):
            
            if z_distance(x02, m, y0_2, l_b, tubes_zs_b[i], tubes_starts_b[i], tubes_directions_b[i], z_magnet=self.z_magnet) < self.inlier_treshold:
                
                mask_b2.append(0)
                
            else:
                
                mask_b2.append(1)
                
        mask_b2 = np.array(mask_b2, dtype=bool)
        
        mask_a2 = []
                
        for i in range(len(tubes_starts_a)):
            
            if z_distance(x02, m, y0_2, l_a, tubes_zs_a[i], tubes_starts_a[i], tubes_directions_a[i], z_magnet=self.z_magnet) < self.inlier_treshold:
                
                mask_a2.append(0)
                
            else:
                
                mask_a2.append(1)
                
        mask_a2 = np.array(mask_a2, dtype=bool)
                
        labels_b = (-mask_b + 0) * 2 + (-mask_b2 + 0) - 1
        labels_a = (-mask_a + 0) * 2 + (-mask_a2 + 0) - 1
        labels_a[labels_a==2] -= 1
        labels_b[labels_b==2] -= 1
        self.labels_ = [labels_b, labels_a]
        
        m1 = best_model
        m2 = best_model2
        self.tracks_params_ = np.array([[[[m1[2], y0 - self.z_magnet * m1[2]], [m1[1], m1[0] - self.z_magnet * m1[1]]],\
                                        [[m2[2], y0_2 - self.z_magnet * m2[2]], [m2[1], m2[0] - self.z_magnet * m2[1]]]],\
                                        [[[m1[3], y0 - self.z_magnet * m1[3]], [m1[1], m1[0] - self.z_magnet * m1[1]]],\
                                         [[m2[3], y0_2 - self.z_magnet * m2[3]], [m2[1], m2[0] - self.z_magnet * m2[1]]]]])
        '''
        scaler = Scaler(z_scale=self.z_scale, y_scale=self.y_scale, x_scale=self.x_scale)
        self.z_magnet = self.z_magnet / self.z_scale
        
        normed_ends = scaler.transform(ends_of_strawtubes_b)
        
        starts = []
        directions = []
        z0s = []
        
        for i in range(len(normed_ends)):
            
            start, direction, z0 = ends2params(normed_ends[i])
            starts.append(start)
            directions.append(direction)
            z0s.append(z0)
        
        tubes_starts_b = np.array(starts)
        tubes_directions_b = np.array(directions)
        tubes_zs_b = np.array(z0s)
        
        normed_ends = scaler.transform(ends_of_strawtubes_a)
        
        starts = []
        directions = []
        z0s = []
        
        for i in range(len(normed_ends)):
            
            start, direction, z0 = ends2params(normed_ends[i])
            starts.append(start)
            directions.append(direction)
            z0s.append(z0)
        
        tubes_starts_a = np.array(starts)
        tubes_directions_a = np.array(directions)
        tubes_zs_a = np.array(z0s)
        
        start = scaler.parameters_transform(np.array([x0, 0, 0, 0]))
        dots, values = self.conjugate_gradient_method(start, y0, tubes_starts_b, tubes_directions_b,\
                                                                                  tubes_zs_b, dists_b, tubes_starts_a,\
                                                                                  tubes_directions_a, tubes_zs_a, dists_a)
        return scaler.parameters_inverse_transform(dots), values, y0
        '''
        
        #return [[best_model], [best_model2]], [y0, y0_2]