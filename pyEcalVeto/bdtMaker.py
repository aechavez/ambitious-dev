import argparse
import logging
import matplotlib as mpl
import numpy as np
import os
import pickle as pkl
import ROOT as r
import sys
import xgboost as xgb


# Load dependencies
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
mpl.use('Agg')


################################
# Class to hold events
################################

class event_container:

    def __init__(self, file_name, max_events, training_fraction, is_signal, print_frequency = 1000):

        print('\n[ INFO ] - Building event container')

        self.tree = r.TChain('EcalVeto')
        self.tree.Add(file_name)
        self.max_events = max_events
        self.training_fraction = training_fraction
        self.is_signal = is_signal
        self.events = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.event_count = 0
        self.print_frequency = print_frequency

    # Function to convert events from ROOT to Python
    def process_events(self, print_frequency = 1000):

        # Reset some attributes if desired
        if print_frequency != self.print_frequency: self.print_frequency = print_frequency

        self.events = []

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
                event.straight4                         ,
                event.firstNearPhotonLayer              ,
                event.nNearPhotonHits                   ,
                event.fullElectronTerritoryHits         ,
                event.fullPhotonTerritoryHits           ,
                event.fullTerritoryRatio                ,
                event.electronTerritoryHits             ,
                event.photonTerritoryHits               ,
                event.territoryRatio                    ,
                event.trajectorySeparation              ,
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

    # Function to split events for training and cross-validation
    def train_test_split(self):

        self.x_train = self.events[:int(self.training_fraction*self.events.shape[0])]
        self.y_train = np.zeros(self.x_train.shape[0]) + (self.is_signal == True)

        self.x_test = self.events[int(self.training_fraction*self.events.shape[0]):]
        self.y_test = np.zeros(self.x_test.shape[0]) + (self.is_signal == True)


######################################################
# Class to hold signal and background events
######################################################

class merged_event_container:

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

    # Parse arguments passed by the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, action = 'store', dest = 'seed', default = 2,
                        help = 'Seed for NumPy randomness')
    parser.add_argument('--train_frac', type = float, action = 'store', dest = 'training_fraction', default = 0.8,
                        help = 'Fraction of events to use for training')
    parser.add_argument('--eta', type = float, action = 'store', dest = 'eta', default = 0.023,
                        help = 'Learning rate')
    parser.add_argument('--num_rounds', type = int, action = 'store', dest = 'num_rounds', default = 1000,
                        help = 'Number of boosting rounds')
    parser.add_argument('--max_depth', type = int, action = 'store', dest = 'max_depth', default = 10,
                        help = 'Maximum tree depth')
    parser.add_argument('-s', action = 'store', dest = 'signal_file',
                        help = 'Signal file name')
    parser.add_argument('-b', action = 'store', dest = 'background_file',
                        help = 'Background file name')
    parser.add_argument('-o', action = 'store', dest = 'out_name', default = 'bdt',
                        help = 'Output file name')
    parser.add_argument('-m', type = int, action = 'store', dest = 'max_events', default = 1500000,
                        help = 'Maximum number of events to run over')
    args = parser.parse_args()

    # Seed NumPy randomness
    np.random.seed(args.seed)

    # Get BDT num
    bdt_num=0
    Check=True
    while Check:
        if not os.path.exists(options.out_name+'_'+str(bdt_num)):
            try:
                os.makedirs(options.out_name+'_'+str(bdt_num))
                Check=False
            except:
               Check=True
        else:
            bdt_num+=1

    # Print run info
    print( 'Random seed is = {}'.format(options.seed)             )
    print( 'You set max_evt = {}'.format(options.max_evt)         )
    print( 'You set tree number = {}'.format(options.tree_number) )
    print( 'You set max tree depth = {}'.format(options.depth)    )
    print( 'You set eta = {}'.format(options.eta)                 )

    # Make Signal Container
    print( 'Loading sig_file = {}'.format(options.sig_file) )
    sigContainer = sampleContainer(options.sig_file,options.max_evt,options.train_frac,True)
    sigContainer.root2PyEvents()
    sigContainer.constructTrainAndTest()

    # Make Background Container
    print( 'Loading bkg_file = {}'.format(options.bkg_file) )
    bkgContainer = sampleContainer(options.bkg_file,options.max_evt,options.train_frac,False)
    bkgContainer.root2PyEvents()
    bkgContainer.constructTrainAndTest()

    # Merge
    eventContainer = mergedContainer(sigContainer,bkgContainer)

    params = {
               'objective': 'binary:logistic',
               'eta': options.eta,
               'max_depth': options.depth,
               'min_child_weight': 20,
               # 'silent': 1,
               'subsample':.9,
               'colsample_bytree': .85,
               # 'eval_metric': 'auc',
               'eval_metric': 'error',
               'seed': 1,
               'nthread': 1,
               'verbosity': 1
               # 'early_stopping_rounds' : 10
    }

    # Train the BDT model
    evallist = [(eventContainer.dtest,'eval'), (eventContainer.dtrain,'train')]
    gbm = xgb.train(params, eventContainer.dtrain, options.tree_number, evallist, early_stopping_rounds = 10)

    # Store BDT
    output = open(options.out_name+'_'+str(bdt_num)+'/' + \
            options.out_name+'_'+str(bdt_num)+'_weights.pkl', 'wb')
    pkl.dump(gbm, output)

    # Plot feature importances
    xgb.plot_importance(gbm)
    plt.pyplot.savefig(options.out_name+'_'+str(bdt_num)+"/" + \
            options.out_name+'_'+str(bdt_num)+'_fimportance.png', # png file name
            dpi=500, bbox_inches='tight', pad_inches=0.5) # png parameters
    
    # Closing statment
    print("Files saved in: ", options.out_name+'_'+str(bdt_num))


###############
# RUN
###############

if __name__ == '__main__':
    main()
