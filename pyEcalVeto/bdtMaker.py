import argparse
import logging
import matplotlib as mpl
import numpy as np
import os
import pickle as pkl
import ROOT as r
import xgboost as xgb


# Load dependencies
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
mpl.use('Agg')


################################
# Class to hold events
################################

class EventContainer:

    def __init__(self, file_name, is_signal, tree_name = 'EcalVeto', max_events = -1, training_fraction = 0.8,
                 print_frequency = 1000):

        self.file_name = file_name
        self.is_signal = is_signal
        self.tree_name = tree_name
        self.tree = r.TChain(self.tree_name)
        self.tree.Add(self.file_name)
        self.max_events = max_events
        self.event_count = 0
        self.training_fraction = training_fraction
        self.print_frequency = print_frequency
        self.events = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    # Method to build the event container
    def build(self, max_events = -1, print_frequency = 1000):

        if self.is_signal: print('\n[ INFO ] - Building signal event container')
        else: print('\n[ INFO ] - Building background event container')

        # Reset some attributes if desired
        if max_events != self.max_events: self.max_events = max_events
        if print_frequency != self.print_frequency: self.print_frequency = print_frequency

        # max_events should be between 1 and tree->GetEntries()
        if self.max_events < 1 or self.max_events > self.tree.GetEntries():
            self.max_events = self.tree.GetEntries()

        # Loop to process each event
        for event in self.tree:

            # Stop the loop if it reaches the max
            if self.event_count >= self.max_events: break

            if self.event_count%self.print_frequency == 0:
                print('Processing event: {}'.format(self.event_count))

            evt = [

                # Fernand variables
                event.nReadoutHits                      ,
                event.summedDet                         ,
                event.summedTightIso                    ,
                event.maxCellDep                        ,
                event.showerRMS                         ,
                event.xStd                              ,
                event.yStd                              ,
                event.avgLayerHit                       ,
                event.stdLayerHit                       ,
                event.deepestLayerHit                   ,
                event.ecalBackEnergy                    ,

                # MIP tracking variables
                event.nStraightTracks                   ,
                event.firstNearPhotonLayer              ,
                event.nNearPhotonHits                   ,
                event.nFullElectronTerritoryHits        ,
                event.nFullPhotonTerritoryHits          ,
                event.fullTerritoryRatio                ,
                event.nElectronTerritoryHits            ,
                event.nPhotonTerritoryHits              ,
                event.territoryRatio                    ,
                event.trajectorySep                     ,
                event.trajectoryDot                     ,

                # Longitudinal segment variables
                event.energy_s1                         ,
                event.nHits_s1                          ,
                event.xMean_s1                          ,
                event.yMean_s1                          ,
                event.layerMean_s1                      ,
                event.xStd_s1                           ,
                event.yStd_s1                           ,
                event.layerStd_s1                       ,

                event.energy_s2                         ,
                event.nHits_s2                          ,
                event.xMean_s2                          ,
                event.yMean_s2                          ,
                event.layerMean_s2                      ,
                event.xStd_s2                           ,
                event.yStd_s2                           ,
                event.layerStd_s2                       ,

                event.energy_s3                         ,
                event.nHits_s3                          ,
                event.xMean_s3                          ,
                event.yMean_s3                          ,
                event.layerMean_s3                      ,
                event.xStd_s3                           ,
                event.yStd_s3                           ,
                event.layerStd_s3                       ,

                # Electron RoC variables
                event.electronContainmentEnergy_x1_s1   ,
                event.electronContainmentEnergy_x2_s1   ,
                event.electronContainmentEnergy_x3_s1   ,
                event.electronContainmentEnergy_x4_s1   ,
                event.electronContainmentEnergy_x5_s1   ,

                event.electronContainmentNHits_x1_s1    ,
                event.electronContainmentNHits_x2_s1    ,
                event.electronContainmentNHits_x3_s1    ,
                event.electronContainmentNHits_x4_s1    ,
                event.electronContainmentNHits_x5_s1    ,

                event.electronContainmentXMean_x1_s1    ,
                event.electronContainmentXMean_x2_s1    ,
                event.electronContainmentXMean_x3_s1    ,
                event.electronContainmentXMean_x4_s1    ,
                event.electronContainmentXMean_x5_s1    ,

                event.electronContainmentYMean_x1_s1    ,
                event.electronContainmentYMean_x2_s1    ,
                event.electronContainmentYMean_x3_s1    ,
                event.electronContainmentYMean_x4_s1    ,
                event.electronContainmentYMean_x5_s1    ,

                event.electronContainmentLayerMean_x1_s1,
                event.electronContainmentLayerMean_x2_s1,
                event.electronContainmentLayerMean_x3_s1,
                event.electronContainmentLayerMean_x4_s1,
                event.electronContainmentLayerMean_x5_s1,

                event.electronContainmentXStd_x1_s1     ,
                event.electronContainmentXStd_x2_s1     ,
                event.electronContainmentXStd_x3_s1     ,
                event.electronContainmentXStd_x4_s1     ,
                event.electronContainmentXStd_x5_s1     ,

                event.electronContainmentYStd_x1_s1     ,
                event.electronContainmentYStd_x2_s1     ,
                event.electronContainmentYStd_x3_s1     ,
                event.electronContainmentYStd_x4_s1     ,
                event.electronContainmentYStd_x5_s1     ,

                event.electronContainmentLayerStd_x1_s1 ,
                event.electronContainmentLayerStd_x2_s1 ,
                event.electronContainmentLayerStd_x3_s1 ,
                event.electronContainmentLayerStd_x4_s1 ,
                event.electronContainmentLayerStd_x5_s1 ,

                event.electronContainmentEnergy_x1_s2   ,
                event.electronContainmentEnergy_x2_s2   ,
                event.electronContainmentEnergy_x3_s2   ,
                event.electronContainmentEnergy_x4_s2   ,
                event.electronContainmentEnergy_x5_s2   ,

                event.electronContainmentNHits_x1_s2    ,
                event.electronContainmentNHits_x2_s2    ,
                event.electronContainmentNHits_x3_s2    ,
                event.electronContainmentNHits_x4_s2    ,
                event.electronContainmentNHits_x5_s2    ,

                event.electronContainmentXMean_x1_s2    ,
                event.electronContainmentXMean_x2_s2    ,
                event.electronContainmentXMean_x3_s2    ,
                event.electronContainmentXMean_x4_s2    ,
                event.electronContainmentXMean_x5_s2    ,

                event.electronContainmentYMean_x1_s2    ,
                event.electronContainmentYMean_x2_s2    ,
                event.electronContainmentYMean_x3_s2    ,
                event.electronContainmentYMean_x4_s2    ,
                event.electronContainmentYMean_x5_s2    ,

                event.electronContainmentLayerMean_x1_s2,
                event.electronContainmentLayerMean_x2_s2,
                event.electronContainmentLayerMean_x3_s2,
                event.electronContainmentLayerMean_x4_s2,
                event.electronContainmentLayerMean_x5_s2,

                event.electronContainmentXStd_x1_s2     ,
                event.electronContainmentXStd_x2_s2     ,
                event.electronContainmentXStd_x3_s2     ,
                event.electronContainmentXStd_x4_s2     ,
                event.electronContainmentXStd_x5_s2     ,

                event.electronContainmentYStd_x1_s2     ,
                event.electronContainmentYStd_x2_s2     ,
                event.electronContainmentYStd_x3_s2     ,
                event.electronContainmentYStd_x4_s2     ,
                event.electronContainmentYStd_x5_s2     ,

                event.electronContainmentLayerStd_x1_s2 ,
                event.electronContainmentLayerStd_x2_s2 ,
                event.electronContainmentLayerStd_x3_s2 ,
                event.electronContainmentLayerStd_x4_s2 ,
                event.electronContainmentLayerStd_x5_s2 ,

                event.electronContainmentEnergy_x1_s3   ,
                event.electronContainmentEnergy_x2_s3   ,
                event.electronContainmentEnergy_x3_s3   ,
                event.electronContainmentEnergy_x4_s3   ,
                event.electronContainmentEnergy_x5_s3   ,

                event.electronContainmentNHits_x1_s3    ,
                event.electronContainmentNHits_x2_s3    ,
                event.electronContainmentNHits_x3_s3    ,
                event.electronContainmentNHits_x4_s3    ,
                event.electronContainmentNHits_x5_s3    ,

                event.electronContainmentXMean_x1_s3    ,
                event.electronContainmentXMean_x2_s3    ,
                event.electronContainmentXMean_x3_s3    ,
                event.electronContainmentXMean_x4_s3    ,
                event.electronContainmentXMean_x5_s3    ,

                event.electronContainmentYMean_x1_s3    ,
                event.electronContainmentYMean_x2_s3    ,
                event.electronContainmentYMean_x3_s3    ,
                event.electronContainmentYMean_x4_s3    ,
                event.electronContainmentYMean_x5_s3    ,

                event.electronContainmentLayerMean_x1_s3,
                event.electronContainmentLayerMean_x2_s3,
                event.electronContainmentLayerMean_x3_s3,
                event.electronContainmentLayerMean_x4_s3,
                event.electronContainmentLayerMean_x5_s3,

                event.electronContainmentXStd_x1_s3     ,
                event.electronContainmentXStd_x2_s3     ,
                event.electronContainmentXStd_x3_s3     ,
                event.electronContainmentXStd_x4_s3     ,
                event.electronContainmentXStd_x5_s3     ,

                event.electronContainmentYStd_x1_s3     ,
                event.electronContainmentYStd_x2_s3     ,
                event.electronContainmentYStd_x3_s3     ,
                event.electronContainmentYStd_x4_s3     ,
                event.electronContainmentYStd_x5_s3     ,

                event.electronContainmentLayerStd_x1_s3 ,
                event.electronContainmentLayerStd_x2_s3 ,
                event.electronContainmentLayerStd_x3_s3 ,
                event.electronContainmentLayerStd_x4_s3 ,
                event.electronContainmentLayerStd_x5_s3 ,

                # Photon RoC variables
                event.photonContainmentEnergy_x1_s1     ,
                event.photonContainmentEnergy_x2_s1     ,
                event.photonContainmentEnergy_x3_s1     ,
                event.photonContainmentEnergy_x4_s1     ,
                event.photonContainmentEnergy_x5_s1     ,

                event.photonContainmentNHits_x1_s1      ,
                event.photonContainmentNHits_x2_s1      ,
                event.photonContainmentNHits_x3_s1      ,
                event.photonContainmentNHits_x4_s1      ,
                event.photonContainmentNHits_x5_s1      ,

                event.photonContainmentXMean_x1_s1      ,
                event.photonContainmentXMean_x2_s1      ,
                event.photonContainmentXMean_x3_s1      ,
                event.photonContainmentXMean_x4_s1      ,
                event.photonContainmentXMean_x5_s1      ,

                event.photonContainmentYMean_x1_s1      ,
                event.photonContainmentYMean_x2_s1      ,
                event.photonContainmentYMean_x3_s1      ,
                event.photonContainmentYMean_x4_s1      ,
                event.photonContainmentYMean_x5_s1      ,

                event.photonContainmentLayerMean_x1_s1  ,
                event.photonContainmentLayerMean_x2_s1  ,
                event.photonContainmentLayerMean_x3_s1  ,
                event.photonContainmentLayerMean_x4_s1  ,
                event.photonContainmentLayerMean_x5_s1  ,

                event.photonContainmentXStd_x1_s1       ,
                event.photonContainmentXStd_x2_s1       ,
                event.photonContainmentXStd_x3_s1       ,
                event.photonContainmentXStd_x4_s1       ,
                event.photonContainmentXStd_x5_s1       ,

                event.photonContainmentYStd_x1_s1       ,
                event.photonContainmentYStd_x2_s1       ,
                event.photonContainmentYStd_x3_s1       ,
                event.photonContainmentYStd_x4_s1       ,
                event.photonContainmentYStd_x5_s1       ,

                event.photonContainmentLayerStd_x1_s1   ,
                event.photonContainmentLayerStd_x2_s1   ,
                event.photonContainmentLayerStd_x3_s1   ,
                event.photonContainmentLayerStd_x4_s1   ,
                event.photonContainmentLayerStd_x5_s1   ,

                event.photonContainmentEnergy_x1_s2     ,
                event.photonContainmentEnergy_x2_s2     ,
                event.photonContainmentEnergy_x3_s2     ,
                event.photonContainmentEnergy_x4_s2     ,
                event.photonContainmentEnergy_x5_s2     ,

                event.photonContainmentNHits_x1_s2      ,
                event.photonContainmentNHits_x2_s2      ,
                event.photonContainmentNHits_x3_s2      ,
                event.photonContainmentNHits_x4_s2      ,
                event.photonContainmentNHits_x5_s2      ,

                event.photonContainmentXMean_x1_s2      ,
                event.photonContainmentXMean_x2_s2      ,
                event.photonContainmentXMean_x3_s2      ,
                event.photonContainmentXMean_x4_s2      ,
                event.photonContainmentXMean_x5_s2      ,

                event.photonContainmentYMean_x1_s2      ,
                event.photonContainmentYMean_x2_s2      ,
                event.photonContainmentYMean_x3_s2      ,
                event.photonContainmentYMean_x4_s2      ,
                event.photonContainmentYMean_x5_s2      ,

                event.photonContainmentLayerMean_x1_s2  ,
                event.photonContainmentLayerMean_x2_s2  ,
                event.photonContainmentLayerMean_x3_s2  ,
                event.photonContainmentLayerMean_x4_s2  ,
                event.photonContainmentLayerMean_x5_s2  ,

                event.photonContainmentXStd_x1_s2       ,
                event.photonContainmentXStd_x2_s2       ,
                event.photonContainmentXStd_x3_s2       ,
                event.photonContainmentXStd_x4_s2       ,
                event.photonContainmentXStd_x5_s2       ,

                event.photonContainmentYStd_x1_s2       ,
                event.photonContainmentYStd_x2_s2       ,
                event.photonContainmentYStd_x3_s2       ,
                event.photonContainmentYStd_x4_s2       ,
                event.photonContainmentYStd_x5_s2       ,

                event.photonContainmentLayerStd_x1_s2   ,
                event.photonContainmentLayerStd_x2_s2   ,
                event.photonContainmentLayerStd_x3_s2   ,
                event.photonContainmentLayerStd_x4_s2   ,
                event.photonContainmentLayerStd_x5_s2   ,

                event.photonContainmentEnergy_x1_s3     ,
                event.photonContainmentEnergy_x2_s3     ,
                event.photonContainmentEnergy_x3_s3     ,
                event.photonContainmentEnergy_x4_s3     ,
                event.photonContainmentEnergy_x5_s3     ,

                event.photonContainmentNHits_x1_s3      ,
                event.photonContainmentNHits_x2_s3      ,
                event.photonContainmentNHits_x3_s3      ,
                event.photonContainmentNHits_x4_s3      ,
                event.photonContainmentNHits_x5_s3      ,

                event.photonContainmentXMean_x1_s3      ,
                event.photonContainmentXMean_x2_s3      ,
                event.photonContainmentXMean_x3_s3      ,
                event.photonContainmentXMean_x4_s3      ,
                event.photonContainmentXMean_x5_s3      ,

                event.photonContainmentYMean_x1_s3      ,
                event.photonContainmentYMean_x2_s3      ,
                event.photonContainmentYMean_x3_s3      ,
                event.photonContainmentYMean_x4_s3      ,
                event.photonContainmentYMean_x5_s3      ,

                event.photonContainmentLayerMean_x1_s3  ,
                event.photonContainmentLayerMean_x2_s3  ,
                event.photonContainmentLayerMean_x3_s3  ,
                event.photonContainmentLayerMean_x4_s3  ,
                event.photonContainmentLayerMean_x5_s3  ,

                event.photonContainmentXStd_x1_s3       ,
                event.photonContainmentXStd_x2_s3       ,
                event.photonContainmentXStd_x3_s3       ,
                event.photonContainmentXStd_x4_s3       ,
                event.photonContainmentXStd_x5_s3       ,

                event.photonContainmentYStd_x1_s3       ,
                event.photonContainmentYStd_x2_s3       ,
                event.photonContainmentYStd_x3_s3       ,
                event.photonContainmentYStd_x4_s3       ,
                event.photonContainmentYStd_x5_s3       ,

                event.photonContainmentLayerStd_x1_s3   ,
                event.photonContainmentLayerStd_x2_s3   ,
                event.photonContainmentLayerStd_x3_s3   ,
                event.photonContainmentLayerStd_x4_s3   ,
                event.photonContainmentLayerStd_x5_s3   ,

                # Outside RoC variables
                event.outsideContainmentEnergy_x1_s1    ,
                event.outsideContainmentEnergy_x2_s1    ,
                event.outsideContainmentEnergy_x3_s1    ,
                event.outsideContainmentEnergy_x4_s1    ,
                event.outsideContainmentEnergy_x5_s1    ,

                event.outsideContainmentNHits_x1_s1     ,
                event.outsideContainmentNHits_x2_s1     ,
                event.outsideContainmentNHits_x3_s1     ,
                event.outsideContainmentNHits_x4_s1     ,
                event.outsideContainmentNHits_x5_s1     ,

                event.outsideContainmentXMean_x1_s1     ,
                event.outsideContainmentXMean_x2_s1     ,
                event.outsideContainmentXMean_x3_s1     ,
                event.outsideContainmentXMean_x4_s1     ,
                event.outsideContainmentXMean_x5_s1     ,

                event.outsideContainmentYMean_x1_s1     ,
                event.outsideContainmentYMean_x2_s1     ,
                event.outsideContainmentYMean_x3_s1     ,
                event.outsideContainmentYMean_x4_s1     ,
                event.outsideContainmentYMean_x5_s1     ,

                event.outsideContainmentLayerMean_x1_s1 ,
                event.outsideContainmentLayerMean_x2_s1 ,
                event.outsideContainmentLayerMean_x3_s1 ,
                event.outsideContainmentLayerMean_x4_s1 ,
                event.outsideContainmentLayerMean_x5_s1 ,

                event.outsideContainmentXStd_x1_s1      ,
                event.outsideContainmentXStd_x2_s1      ,
                event.outsideContainmentXStd_x3_s1      ,
                event.outsideContainmentXStd_x4_s1      ,
                event.outsideContainmentXStd_x5_s1      ,

                event.outsideContainmentYStd_x1_s1      ,
                event.outsideContainmentYStd_x2_s1      ,
                event.outsideContainmentYStd_x3_s1      ,
                event.outsideContainmentYStd_x4_s1      ,
                event.outsideContainmentYStd_x5_s1      ,

                event.outsideContainmentLayerStd_x1_s1  ,
                event.outsideContainmentLayerStd_x2_s1  ,
                event.outsideContainmentLayerStd_x3_s1  ,
                event.outsideContainmentLayerStd_x4_s1  ,
                event.outsideContainmentLayerStd_x5_s1  ,

                event.outsideContainmentEnergy_x1_s2    ,
                event.outsideContainmentEnergy_x2_s2    ,
                event.outsideContainmentEnergy_x3_s2    ,
                event.outsideContainmentEnergy_x4_s2    ,
                event.outsideContainmentEnergy_x5_s2    ,

                event.outsideContainmentNHits_x1_s2     ,
                event.outsideContainmentNHits_x2_s2     ,
                event.outsideContainmentNHits_x3_s2     ,
                event.outsideContainmentNHits_x4_s2     ,
                event.outsideContainmentNHits_x5_s2     ,

                event.outsideContainmentXMean_x1_s2     ,
                event.outsideContainmentXMean_x2_s2     ,
                event.outsideContainmentXMean_x3_s2     ,
                event.outsideContainmentXMean_x4_s2     ,
                event.outsideContainmentXMean_x5_s2     ,

                event.outsideContainmentYMean_x1_s2     ,
                event.outsideContainmentYMean_x2_s2     ,
                event.outsideContainmentYMean_x3_s2     ,
                event.outsideContainmentYMean_x4_s2     ,
                event.outsideContainmentYMean_x5_s2     ,

                event.outsideContainmentLayerMean_x1_s2 ,
                event.outsideContainmentLayerMean_x2_s2 ,
                event.outsideContainmentLayerMean_x3_s2 ,
                event.outsideContainmentLayerMean_x4_s2 ,
                event.outsideContainmentLayerMean_x5_s2 ,

                event.outsideContainmentXStd_x1_s2      ,
                event.outsideContainmentXStd_x2_s2      ,
                event.outsideContainmentXStd_x3_s2      ,
                event.outsideContainmentXStd_x4_s2      ,
                event.outsideContainmentXStd_x5_s2      ,

                event.outsideContainmentYStd_x1_s2      ,
                event.outsideContainmentYStd_x2_s2      ,
                event.outsideContainmentYStd_x3_s2      ,
                event.outsideContainmentYStd_x4_s2      ,
                event.outsideContainmentYStd_x5_s2      ,

                event.outsideContainmentLayerStd_x1_s2  ,
                event.outsideContainmentLayerStd_x2_s2  ,
                event.outsideContainmentLayerStd_x3_s2  ,
                event.outsideContainmentLayerStd_x4_s2  ,
                event.outsideContainmentLayerStd_x5_s2  ,

                event.outsideContainmentEnergy_x1_s3    ,
                event.outsideContainmentEnergy_x2_s3    ,
                event.outsideContainmentEnergy_x3_s3    ,
                event.outsideContainmentEnergy_x4_s3    ,
                event.outsideContainmentEnergy_x5_s3    ,

                event.outsideContainmentNHits_x1_s3     ,
                event.outsideContainmentNHits_x2_s3     ,
                event.outsideContainmentNHits_x3_s3     ,
                event.outsideContainmentNHits_x4_s3     ,
                event.outsideContainmentNHits_x5_s3     ,

                event.outsideContainmentXMean_x1_s3     ,
                event.outsideContainmentXMean_x2_s3     ,
                event.outsideContainmentXMean_x3_s3     ,
                event.outsideContainmentXMean_x4_s3     ,
                event.outsideContainmentXMean_x5_s3     ,

                event.outsideContainmentYMean_x1_s3     ,
                event.outsideContainmentYMean_x2_s3     ,
                event.outsideContainmentYMean_x3_s3     ,
                event.outsideContainmentYMean_x4_s3     ,
                event.outsideContainmentYMean_x5_s3     ,

                event.outsideContainmentLayerMean_x1_s3 ,
                event.outsideContainmentLayerMean_x2_s3 ,
                event.outsideContainmentLayerMean_x3_s3 ,
                event.outsideContainmentLayerMean_x4_s3 ,
                event.outsideContainmentLayerMean_x5_s3 ,

                event.outsideContainmentXStd_x1_s3      ,
                event.outsideContainmentXStd_x2_s3      ,
                event.outsideContainmentXStd_x3_s3      ,
                event.outsideContainmentXStd_x4_s3      ,
                event.outsideContainmentXStd_x5_s3      ,

                event.outsideContainmentYStd_x1_s3      ,
                event.outsideContainmentYStd_x2_s3      ,
                event.outsideContainmentYStd_x3_s3      ,
                event.outsideContainmentYStd_x4_s3      ,
                event.outsideContainmentYStd_x5_s3      ,

                event.outsideContainmentLayerStd_x1_s3  ,
                event.outsideContainmentLayerStd_x2_s3  ,
                event.outsideContainmentLayerStd_x3_s3  ,
                event.outsideContainmentLayerStd_x4_s3  ,
                event.outsideContainmentLayerStd_x5_s3
            ]

            self.events.append(evt)
            self.event_count += 1

        # Convert the list to an array and scramble
        self.events = np.array(self.events)
        new_idx = np.random.permutation(np.arange(self.events.shape[0]))
        np.take(self.events, new_idx, axis = 0, out = self.events)

        print('\n[ INFO ] - Container shape: {}'.format(self.events.shape))

    # Method to split events for training and cross-validation
    def train_test_split(self):

        self.x_train = self.events[:int(self.training_fraction*self.events.shape[0])]
        self.y_train = np.zeros(self.x_train.shape[0]) + (self.is_signal == True)

        self.x_test = self.events[int(self.training_fraction*self.events.shape[0]):]
        self.y_test = np.zeros(self.x_test.shape[0]) + (self.is_signal == True)


######################################################
# Class to hold signal and background events
######################################################

class MergedEventContainer:

    def __init__(self, signal_container, background_container):

        self.x_train = np.vstack((signal_container.x_train, background_container.x_train))
        self.y_train = np.append(signal_container.y_train, background_container.y_train)
        self.d_train = xgb.DMatrix(self.x_train, self.y_train)

        self.x_test = np.vstack((signal_container.x_test, background_container.x_test))
        self.y_test = np.append(signal_container.y_test, background_container.y_test)
        self.d_test = xgb.DMatrix(self.x_test, self.y_test)


###########################
# Main subroutine
###########################

def main():

    # Set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, action = 'store', dest = 'seed', default = 0,
                        help = 'Seed for reproducibility (Default: 0)')
    parser.add_argument('--train_frac', type = float, action = 'store', dest = 'training_fraction', default = 0.8,
                        help = 'Fraction of events to use for training (Default: 0.8)')
    parser.add_argument('--num_boost_round', type = int, action = 'store', dest = 'boosting_rounds', default = 1000,
                        help = 'Number of boosting rounds (Default: 1000)')
    parser.add_argument('--early_stopping_rounds', type = int, action = 'store', dest = 'stopping_rounds', default = 10,
                        help = 'Number of early stopping rounds (Default: 10)')
    parser.add_argument('--eta', type = float, action = 'store', dest = 'learning_rate', default = 0.023,
                        help = 'Learning rate (Default: 0.023)')
    parser.add_argument('--max_depth', type = int, action = 'store', dest = 'max_depth', default = 10,
                        help = 'Maximum tree depth (Default: 10)')
    parser.add_argument('--min_child_weight', type = float, action = 'store', dest = 'min_child_weight', default = 20.,
                        help = 'Minimum child weight (Default: 20)')
    parser.add_argument('--subsample', type = float, action = 'store', dest = 'row_subsample_ratio', default = 0.9,
                        help = 'Row subsampling ratio (Default: 0.9)')
    parser.add_argument('--colsample_bytree', type = float, action = 'store', dest = 'column_subsample_ratio', default = 0.85,
                        help = 'Column subsampling ratio (Default: 0.85)')
    parser.add_argument('-s', action = 'store', dest = 'signal_file',
                        help = 'Signal file')
    parser.add_argument('-b', action = 'store', dest = 'background_file',
                        help = 'Background file')
    parser.add_argument('-o', action = 'store', dest = 'model_name', default = 'bdt',
                        help = 'Name of the BDT model to train')
    parser.add_argument('-m', type = int, action = 'store', dest = 'max_events', default = -1,
                        help = 'Maximum number of events to run on')
    args = parser.parse_args()

    # Seed for reproducibility
    np.random.seed(args.seed)

    # Assign a number label to this training session
    num = 0
    check = True
    while check:
        if os.path.exists('{}_train_out_{}'.format(args.model_name, num)): num += 1
        else: check = False

    # Make the output directory
    out_directory = '{}_train_out_{}'.format(args.model_name, num)
    print('\n[ INFO ] - Making output directory: {}'.format(out_directory))
    os.makedirs(out_directory)

    # Print settings for this session
    print('\n[ INFO ] - You set random seed: {}'.format(args.seed))
    print('[ INFO ] - You set training fraction: {}'.format(args.training_fraction))
    print('[ INFO ] - You set number boosting rounds: {}'.format(args.boosting_rounds))
    print('[ INFO ] - You set number early stopping rounds: {}'.format(args.stopping_rounds))
    print('[ INFO ] - You set learning rate: {}'.format(args.learning_rate))
    print('[ INFO ] - You set maximum tree depth: {}'.format(args.max_depth))
    print('[ INFO ] - You set minimum child weight: {}'.format(args.min_child_weight))
    print('[ INFO ] - You set row subsampling ratio: {}'.format(args.row_subsample_ratio))
    print('[ INFO ] - You set column subsampling ratio: {}'.format(args.column_subsample_ratio))

    # Build the signal container
    signal_container = EventContainer(args.signal_file, True, training_fraction = args.training_fraction)
    signal_container.build(max_events = args.max_events)
    signal_container.train_test_split()

    # Build the background container
    background_container = EventContainer(args.background_file, False, training_fraction = args.training_fraction)
    background_container.build(max_events = args.max_events)
    background_container.train_test_split()

    # Merge the event containers
    merged_container = MergedEventContainer(signal_container, background_container)

    params = {
        'objective': 'binary:logistic',
        'eta': args.learning_rate,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'subsample': args.row_subsample_ratio,
        'colsample_bytree': args.column_subsample_ratio,
        'eval_metric': 'error',
        'seed': args.seed,
        'nthread': 1,
        'verbosity': 1
    }

    # Train the BDT model
    eval_list = [(merged_container.dtrain, 'train'), (merged_container.dtest, 'eval')]
    model = xgb.train(params, merged_container.dtrain, num_boost_round = args.boosting_rounds,
                      evals = eval_list, early_stopping_rounds = args.stopping_rounds)

    # Save the BDT model
    out_file = open('{}/{}_train_out_{}_weights.pkl'.format(out_directory, args.model_name, num), 'wb')
    pkl.dump(model, out_file)

    # Plot feature importance
    xgb.plot_importance(model)
    mpl.pyplot.savefig('{}/{}_train_out_{}_fimportance.png'.format(out_directory, args.model_name, num),
                       dpi = 500, bbox_inches = 'tight', pad_inches = 0.5)

    print('\n[ INFO ] - Training session finished!')


###############
# RUN
###############

if __name__ == '__main__':
    main()
