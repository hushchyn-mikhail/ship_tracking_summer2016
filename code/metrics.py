__author__ = 'mikhail91'

import numpy
import pandas

class HitsMatchingEfficiency(object):

    def __init__(self, eff_threshold=0.5):
        """
        This class calculates tracks efficiencies, reconstruction efficiency, ghost rate and clone rate for one event using hits matching.
        :param eff_threshold: float, threshold value of a track efficiency to consider a track reconstructed.
        :return:
        """

        self.eff_threshold = eff_threshold

    def fit(self, true_labels, labels):
        """
        The method calculates all metrics.
        :param true_labels: numpy.array, true labels of the hits.
        :param labels: numpy.array, recognized labels of the hits.
        :return:
        """

        unique_labels = numpy.unique(labels)

        # Calculate efficiencies
        efficiencies = []
        tracks_id = []

        for lab in unique_labels:

            if lab != -1:

                track = true_labels[labels == lab]
                #if len(track[track != -1]) == 0:
                #    continue
                unique, counts = numpy.unique(track, return_counts=True)

                eff = 1. * counts.max() / len(track)
                efficiencies.append(eff)

                tracks_id.append(unique[counts == counts.max()][0])


        tracks_id = numpy.array(tracks_id)
        efficiencies = numpy.array(efficiencies)
        self.efficiencies_ = efficiencies

        # Calculate avg. efficiency
        avg_efficiency = efficiencies.mean()
        self.avg_efficiency_ = avg_efficiency

        # Calculate reconstruction efficiency
        true_tracks_id = numpy.unique(true_labels)
        n_tracks = (true_tracks_id != -1).sum()

        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        recognition_efficiency = 1. * len(unique) / (n_tracks)
        self.recognition_efficiency_ = recognition_efficiency

        # Calculate ghost rate
        ghost_rate = 1. * (len(tracks_id) - len(reco_tracks_id[reco_tracks_id != -1])) / (n_tracks)
        self.ghost_rate_ = ghost_rate

        # Calculate clone rate
        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        clone_rate = (counts - numpy.ones(len(counts))).sum()/(n_tracks)
        self.clone_rate_ = clone_rate