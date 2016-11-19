__author__ = "oalenkin"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from scipy.optimize import minimize_scalar, basinhopping, differential_evolution
from skimage.draw import line_aa


###############################################################################################################################

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

###############################################################################################################################

def ends2params(array):
    
    start = array[:2]
    k = (array[4]-array[1])/(array[3]-array[0])
    direction = np.array([1., k])/np.sqrt(1+k**2)
    z0 = array[5]
    
    return start, direction, z0

###############################################################################################################################

def adjR(params, *data):
    
    x0, m, y0, l_b, l_a = params
    
    starts_b, perpendiculars_b, zs_b, weights_b, starts_a, perpendiculars_a, zs_a, weights_a, sigma, z_magnet = data
    
    dists_b = np.sum((np.hstack([(x0 + (zs_b - z_magnet) * m).reshape(-1, 1), (y0 + (zs_b - z_magnet) * l_b).reshape(-1, 1)]) -\
                      starts_b) * perpendiculars_b, axis=1)
    
    dists_a = np.sum((np.hstack([(x0 + (zs_a - z_magnet) * m).reshape(-1, 1), (y0 + (zs_a - z_magnet) * l_a).reshape(-1, 1)]) -\
                      starts_a) * perpendiculars_a, axis=1)
    
    dists = np.vstack([dists_a.reshape(-1, 1), dists_b.reshape(-1, 1)]).reshape(-1)
    
    return -np.sum(np.exp((-dists**2)/sigma**2) / np.vstack([weights_a.reshape(-1, 1), weights_b.reshape(-1, 1)]).reshape(-1))

###############################################################################################################################

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

###############################################################################################################################

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
    
###############################################################################################################################

def projection_on_magnet(ends_of_strawtubes, x_resolution=1., y_resolution=1., z_magnet=3070.):
    
    tubes = []
    
    for i in range(len(ends_of_strawtubes)):
        
        tubes.append(ends2params(ends_of_strawtubes[i]))

    lines = []
    z3 = z_magnet
    for i in range(len(tubes)):
        
        for j in range(i+1, len(tubes)):
        
            t1 = tubes[i]
            t2 = tubes[j]
            
            if (t1[2] != t2[2]) and (np.sum(t1[1] - t2[1]) < 0.01):
                
                start1 = t1[0]
                start2 = t2[0]
                z1 = t1[2]
                z2 = t2[2]
                start3 = (start2 - start1) / (z2 - z1) * (z3 - z2) + start2
                
                if (start3[1] > -500) and (start3[1] < 500) and (start3[0] < -225) and (start3[0] > -275):
                
                    lines.append([start3, t1[1]])
    
    x_len = int(600 * x_resolution)
    y_len = int(1200 * y_resolution)
    line_len = 500

    matrix = np.zeros([x_len, y_len])
    
    for line in lines:
        
        rr, cc, val = line_aa(int(round(line[0][0] * x_resolution)) + x_len / 2, int(round(line[0][1] * y_resolution)) + y_len / 2,\
                              int(round((line[0][0] + line[1][0] * line_len) * x_resolution)) + x_len / 2,\
                              int(round((line[0][1] + line[1][1] * line_len) * y_resolution)) + y_len / 2)
        matrix[rr, cc] += 1
        
    return matrix
    
###############################################################################################################################

class RetinaTrackReconstruction(object):
    
    def __init__(self, z_scale=1500., y_scale=1., x_scale=1., z_magnet=3070., noise_treshold=2., inlier_treshold=0.8,\
                 pre_sigma=0.8, x_resolution=1., y_resolution=1., adjusting=True, de_popsize=10, de_mutation=0.9,\
                 de_recombination=0.7, de_seed=42):
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
        self.noise_treshold = noise_treshold
        self.inlier_treshold = inlier_treshold
        self.pre_sigma = pre_sigma
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.z_magnet = z_magnet
        self.adjusting = adjusting
        self.de_popsize = de_popsize
        self.de_recombination= de_recombination
        self.de_mutation = de_mutation
        self.de_seed = de_seed
    
    def get_y0x0(self, ends_of_strawtubes_b, ends_of_strawtubes_a):
        
        projection_b = projection_on_magnet(ends_of_strawtubes_b, self.x_resolution, self.y_resolution, self.z_magnet)
        projection_a = projection_on_magnet(ends_of_strawtubes_a, self.x_resolution, self.y_resolution, self.z_magnet)
        
        projection = projection_b * projection_a
        temp = projection.sum(axis=0)
        y_summary_values = temp
        y_summary_values[:-1] += temp[1:] * 0.3
        y_summary_values[1:] += temp[:-1] * 0.3
        
        y0 = np.argmax(y_summary_values)
        x0 = np.argmax(projection[:, y0])
        
        y0 = (y0 - projection_b.shape[1] * 0.5) / self.y_resolution
        x0 = (x0 - projection_b.shape[0] * 0.5) / self.x_resolution
        if abs(x0) > 450 :
            x0 = 0
        
        return y0, x0
    
    def labeling(self, best_model, y0, best_model2, y0_2, tubes_starts_b, tubes_directions_b, tubes_zs_b,tubes_starts_a,\
                 tubes_directions_a, tubes_zs_a):
        
        x1, m1, l_b1, l_a1 = best_model
        x2, m2, l_b2, l_a2 = best_model2
        
        labels_b = [-1] * len(tubes_starts_b)
        
        for i in range(len(tubes_starts_b)):
            
            z1 = z_distance(x1, m1, y0, l_b1, tubes_zs_b[i], tubes_starts_b[i], tubes_directions_b[i], z_magnet=self.z_magnet)
            z2 = z_distance(x2, m2, y0_2, l_b2, tubes_zs_b[i], tubes_starts_b[i], tubes_directions_b[i], z_magnet=self.z_magnet)
            
            if z1 < np.min([z2, self.inlier_treshold]):
                
                labels_b[i] = 0
                
            elif z2 < np.min([z1, self.inlier_treshold]):
            
                labels_b[i] = 1
                
        labels_b = np.array(labels_b)
        
        labels_a = [-1] * len(tubes_starts_a)
        
        for i in range(len(tubes_starts_a)):
            
            z1 = z_distance(x1, m1, y0, l_a1, tubes_zs_a[i], tubes_starts_a[i], tubes_directions_a[i], z_magnet=self.z_magnet)
            z2 = z_distance(x2, m2, y0_2, l_a2, tubes_zs_a[i], tubes_starts_a[i], tubes_directions_a[i], z_magnet=self.z_magnet)
            
            if z1 < np.min([z2, self.inlier_treshold]):
                
                labels_a[i] = 0
                
            elif z2 < np.min([z1, self.inlier_treshold]):
            
                labels_a[i] = 1
                
        labels_a = np.array(labels_a)
        
        return [labels_b, labels_a]
    
    def adjust(self, track, y0, starts_b, perpendiculars_b, zs_b, weights_b, starts_a, perpendiculars_a, zs_a, weights_a,\
               sigma, dels=[5., 2., 0.005]):
        
        x_del, y_del, ang_del = dels
        
        x0, m, l_b, l_a = track
        
        params = [x0, m, y0, l_b, l_a]
        
        args = (starts_b, perpendiculars_b, zs_b, weights_b, starts_a, perpendiculars_a, zs_a, weights_a, sigma, self.z_magnet)
        
        ang_del = ang_del * self.z_scale
        
        bounds = [(x0-x_del, x0+x_del), (m-ang_del, m+ang_del), (y0-y_del, y0+y_del), (l_b-ang_del, l_b+ang_del),\
                  (l_a-ang_del, l_a+ang_del)]
        
        params = differential_evolution(adjR, args=args, bounds=bounds, popsize=self.de_popsize,\
                                        mutation=self.de_mutation, recombination=self.de_recombination)
        
        p = params.x
        
        return [p[0], p[1], p[3], p[4]], p[2]
         
    def fit(self, event):
        
        ends_of_strawtubes_b = event[event.StatNb < 3][['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz']].values
        ends_of_strawtubes_a = event[event.StatNb > 2][['Wx1', 'Wy1', 'Wz', 'Wx2', 'Wy2', 'Wz']].values
        
        number_of_layer_b = (event[event.StatNb < 3].StatNb.values - 1) * 16 + event[event.StatNb < 3].ViewNb.values * 4 +\
                            event[event.StatNb < 3].PlaneNb.values * 2 + event[event.StatNb < 3].LayerNb.values
        number_of_layer_b = list(number_of_layer_b)
        
        weights_b = np.array([number_of_layer_b.count(i) for i in number_of_layer_b])
        
        number_of_layer_a = (event[event.StatNb > 2].StatNb.values - 1) * 16 + event[event.StatNb > 2].ViewNb.values * 4 +\
                            event[event.StatNb > 2].PlaneNb.values * 2 + event[event.StatNb > 2].LayerNb.values
        number_of_layer_a = list(number_of_layer_a)
        
        weights_a = np.array([number_of_layer_a.count(i) for i in number_of_layer_a])
        
        scaler = Scaler(z_scale=self.z_scale, y_scale=self.y_scale, x_scale=self.x_scale)
        self.z_magnet = self.z_magnet / self.z_scale

        ends_of_strawtubes_b = scaler.transform(ends_of_strawtubes_b)
        
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
        
        ends_of_strawtubes_a = scaler.transform(ends_of_strawtubes_a)
        
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
        
        y0, x0 = self.get_y0x0(ends_of_strawtubes_b, ends_of_strawtubes_a)
        
        best_model = np.array([0, 0, 0, 0])
        
        tubes_perpendiculars_b = np.hstack([-tubes_directions_b[:, 1].reshape(-1, 1), tubes_directions_b[:, 0].reshape(-1, 1)])
        tubes_perpendiculars_a = np.hstack([-tubes_directions_a[:, 1].reshape(-1, 1), tubes_directions_a[:, 0].reshape(-1, 1)])
        best_model, y0 = self.adjust(best_model, y0, tubes_starts_b, tubes_perpendiculars_b, tubes_zs_b, weights_b,\
                                     tubes_starts_a, tubes_perpendiculars_a, tubes_zs_a, weights_a, sigma=self.pre_sigma,\
                                     dels=[250., 2., 0.35])
        
        if self.adjusting:
            tubes_perpendiculars_b = np.hstack([-tubes_directions_b[:, 1].reshape(-1, 1), tubes_directions_b[:, 0].reshape(-1, 1)])
            tubes_perpendiculars_a = np.hstack([-tubes_directions_a[:, 1].reshape(-1, 1), tubes_directions_a[:, 0].reshape(-1, 1)])
            best_model, y0 = self.adjust(best_model, y0, tubes_starts_b, tubes_perpendiculars_b, tubes_zs_b, weights_b,\
                                         tubes_starts_a, tubes_perpendiculars_a, tubes_zs_a, weights_a,\
                                         sigma=self.pre_sigma, dels=[5., 2., 0.005])

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
        
        y0_2, x0_2 = self.get_y0x0(ends_of_strawtubes_b[mask_b], ends_of_strawtubes_a[mask_a])
        
        best_model2 = np.array([0, 0, 0, 0])
        
        tubes_perpendiculars_b = np.hstack([-tubes_directions_b[:, 1].reshape(-1, 1), tubes_directions_b[:, 0].reshape(-1, 1)])
        tubes_perpendiculars_a = np.hstack([-tubes_directions_a[:, 1].reshape(-1, 1), tubes_directions_a[:, 0].reshape(-1, 1)])
        best_model2, y0_2 = self.adjust(best_model2, y0_2, tubes_starts_b[mask_b], tubes_perpendiculars_b[mask_b],\
                                        tubes_zs_b[mask_b], weights_b[mask_b], tubes_starts_a[mask_a],\
                                        tubes_perpendiculars_a[mask_a], tubes_zs_a[mask_a], weights_a[mask_a],\
                                        sigma=self.pre_sigma, dels=[250., 2., 0.35])
            
        if self.adjusting:
            tubes_perpendiculars_b = np.hstack([-tubes_directions_b[:, 1].reshape(-1, 1), tubes_directions_b[:, 0].reshape(-1, 1)])
            tubes_perpendiculars_a = np.hstack([-tubes_directions_a[:, 1].reshape(-1, 1), tubes_directions_a[:, 0].reshape(-1, 1)])
            best_model2, y0_2 = self.adjust(best_model2, y0_2, tubes_starts_b[mask_b], tubes_perpendiculars_b[mask_b],\
                                         tubes_zs_b[mask_b], weights_b[mask_b], tubes_starts_a[mask_a],\
                                            tubes_perpendiculars_a[mask_a], tubes_zs_a[mask_a], weights_a[mask_a],\
                                            sigma=self.pre_sigma, dels=[5., 2., 0.005])
                
        self.labels_ = self.labeling(best_model, y0, best_model2, y0_2, tubes_starts_b, tubes_directions_b, tubes_zs_b,\
                                     tubes_starts_a, tubes_directions_a, tubes_zs_a)
        
        m1 = scaler.parameters_inverse_transform(best_model)
        m2 = scaler.parameters_inverse_transform(best_model2)
        self.z_magnet = self.z_magnet * self.z_scale
        
        self.tracks_params_ = np.array([[[[m1[2], y0 - self.z_magnet * m1[2]], [m1[1], m1[0] - self.z_magnet * m1[1]]],\
                                        [[m2[2], y0_2 - self.z_magnet * m2[2]], [m2[1], m2[0] - self.z_magnet * m2[1]]]],\
                                        [[[m1[3], y0 - self.z_magnet * m1[3]], [m1[1], m1[0] - self.z_magnet * m1[1]]],\
                                         [[m2[3], y0_2 - self.z_magnet * m2[3]], [m2[1], m2[0] - self.z_magnet * m2[1]]]]])
        
        return [[m1], [m2]], [y0, y0_2]