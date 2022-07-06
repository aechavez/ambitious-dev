import numpy as np
import os
import pickle as pkl
import xgboost as xgb
from modules import rootManager as manager


# Load the BDT model
pkl_file = '{}/bdt_train_out_0/bdt_train_out_0_weights.pkl'.format(os.getcwd())
model = pkl.load(open(pkl_file, 'rb'))

# Branch information for tree models
branch_information = {

    # Fernand variables
    'nReadoutHits':               {'dtype': int,   'default': 0 },
    'summedDet':                  {'dtype': float, 'default': 0.},
    'summedTightIso':             {'dtype': float, 'default': 0.},
    'maxCellDep':                 {'dtype': float, 'default': 0.},
    'showerRMS':                  {'dtype': float, 'default': 0.},
    'xStd':                       {'dtype': float, 'default': 0.},
    'yStd':                       {'dtype': float, 'default': 0.},
    'avgLayerHit':                {'dtype': float, 'default': 0.},
    'stdLayerHit':                {'dtype': float, 'default': 0.},
    'deepestLayerHit':            {'dtype': int,   'default': 0 },
    'ecalBackEnergy':             {'dtype': float, 'default': 0.},

    # MIP tracking variables
    'nStraightTracks':            {'dtype': int,   'default': 0 },
    'firstNearPhotonLayer':       {'dtype': int,   'default': 33},
    'nNearPhotonHits':            {'dtype': int,   'default': 0 },
    'nFullElectronTerritoryHits': {'dtype': int,   'default': 0 },
    'nFullPhotonTerritoryHits':   {'dtype': int,   'default': 0 },
    'fullTerritoryRatio':         {'dtype': float, 'default': 1.},
    'nElectronTerritoryHits':     {'dtype': int,   'default': 0 },
    'nPhotonTerritoryHits':       {'dtype': int,   'default': 0 },
    'territoryRatio':             {'dtype': float, 'default': 1.},
    'trajSep':                    {'dtype': float, 'default': 0.},
    'trajDot':                    {'dtype': float, 'default': 0.}
}

for i in range(1, physTools.nsegments + 1):

    # Longitudinal segment variables
    branch_information['energy_s{}'.format(i)]    = {'dtype': float, 'default': 0.}
    branch_information['nHits_s{}'.format(i)]     = {'dtype': int,   'default': 0 }
    branch_information['xMean_s{}'.format(i)]     = {'dtype': float, 'default': 0.}
    branch_information['yMean_s{}'.format(i)]     = {'dtype': float, 'default': 0.}
    branch_information['layerMean_s{}'.format(i)] = {'dtype': float, 'default': 0.}
    branch_information['xStd_s{}'.format(i)]      = {'dtype': float, 'default': 0.}
    branch_information['yStd_s{}'.format(i)]      = {'dtype': float, 'default': 0.}
    branch_information['layerStd_s{}'.format(i)]  = {'dtype': float, 'default': 0.}

    for j in range(1, physTools.nregions + 1):

        # Electron RoC variables
        branch_information['electronContainmentEnergy_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}
        branch_information['electronContainmentNHits_x{}_s{}'.format(j, i)]     = {'dtype': int,   'default': 0 }
        branch_information['electronContainmentXMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['electronContainmentYMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['electronContainmentLayerMean_x{}_s{}'.format(j, i)] = {'dtype': float, 'default': 0.}
        branch_information['electronContainmentXStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['electronContainmentYStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['electronContainmentLayerStd_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}

        # Photon RoC variables
        branch_information['photonContainmentEnergy_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentNHits_x{}_s{}'.format(j, i)]       = {'dtype': int,   'default': 0 }
        branch_information['photonContainmentXMean_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentYMean_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentLayerMean_x{}_s{}'.format(j, i)]   = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentXStd_x{}_s{}'.format(j, i)]        = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentYStd_x{}_s{}'.format(j, i)]        = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentLayerStd_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}

        # Outside RoC variables
        branch_information['outsideContainmentEnergy_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentNHits_x{}_s{}'.format(j, i)]      = {'dtype': int,   'default': 0 }
        branch_information['outsideContainmentXMean_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentYMean_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentLayerMean_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentXStd_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentYStd_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentLayerStd_x{}_s{}'.format(j, i)]   = {'dtype': float, 'default': 0.}

# Needed for BDT analysis
branch_information['discValue'] = {'rtype': float, 'default': 0.5}


###########################
# Main subroutine
###########################

def main():

    # Parse arguments passed by the user
    parsing_dict = manager.parse()
    inputs = parsing_dict['inputs']
    group_labels = parsing_dict['group_labels']
    outputs = parsing_dict['outputs']
    max_events = parsing_dict['max_events']

    # Build a tree process for each file group
    processes = []
    for label, group in zip(group_labels, inputs):
        processes.append(manager.TreeProcess(process_event, file_group = group, name_tag = label))

    # Loop to prepare each process and run
    for process in processes:

        # Move to the temporary directory for this process
        os.chdir(process.temporary_directory)

        # Build a tree model for this process
        process.tree_model = manager.TreeMaker('{}.root'.format(group_labels[processes.index(process)]),\
                                               'EcalVeto', branch_information = branch_information,\
                                               out_directory = outputs[processes.index(process)])

        # Set closing functions
        process.closing_functions = [process.tree_model.write]

        # Run this process
        process.run(max_events = max_events, print_frequency = 100)

    # Remove scratch directory
    manager.remove_scratch()

    print('\n[ INFO ] - All processes finished!')


########################################
# Subroutine to process events
########################################

def process_event(self):

    # Reset branch values for this tree model
    self.tree_model.reset_values()

    # Dictionary of new values
    new_values = {

        # Fernand variables
        'nReadoutHits':                       self.tree.nReadoutHits                      ,
        'summedDet':                          self.tree.summedDet                         ,
        'summedTightIso':                     self.tree.summedTightIso                    ,
        'maxCellDep':                         self.tree.maxCellDep                        ,
        'showerRMS':                          self.tree.showerRMS                         ,
        'xStd':                               self.tree.xStd                              ,
        'yStd':                               self.tree.yStd                              ,
        'avgLayerHit':                        self.tree.avgLayerHit                       ,
        'stdLayerHit':                        self.tree.stdLayerHit                       ,
        'deepestLayerHit':                    self.tree.deepestLayerHit                   ,
        'ecalBackEnergy':                     self.tree.ecalBackEnergy                    ,

        # MIP tracking variables
        'nStraightTracks':                    self.tree.nStraightTracks                   ,
        'firstNearPhotonLayer':               self.tree.firstNearPhotonLayer              ,
        'nNearPhotonHits':                    self.tree.nNearPhotonHits                   ,
        'nFullElectronTerritoryHits':         self.tree.nFullElectronTerritoryHits        ,
        'nFullPhotonTerritoryHits':           self.tree.nFullPhotonTerritoryHits          ,
        'fullTerritoryRatio':                 self.tree.fullTerritoryRatio                ,
        'nElectronTerritoryHits':             self.tree.nElectronTerritoryHits            ,
        'nPhotonTerritoryHits':               self.tree.nPhotonTerritoryHits              ,
        'territoryRatio':                     self.tree.territoryRatio                    ,
        'trajSep':                            self.tree.trajSep                           ,
        'trajDot':                            self.tree.trajDot                           ,

        # Longitudinal segment variables
        'energy_s1':                          self.tree.energy_s1                         ,
        'nHits_s1':                           self.tree.nHits_s1                          ,
        'xMean_s1':                           self.tree.xMean_s1                          ,
        'yMean_s1':                           self.tree.yMean_s1                          ,
        'layerMean_s1':                       self.tree.layerMean_s1                      ,
        'xStd_s1':                            self.tree.xStd_s1                           ,
        'yStd_s1':                            self.tree.yStd_s1                           ,
        'layerStd_s1':                        self.tree.layerStd_s1                       ,

        'energy_s2':                          self.tree.energy_s2                         ,
        'nHits_s2':                           self.tree.nHits_s2                          ,
        'xMean_s2':                           self.tree.xMean_s2                          ,
        'yMean_s2':                           self.tree.yMean_s2                          ,
        'layerMean_s2':                       self.tree.layerMean_s2                      ,
        'xStd_s2':                            self.tree.xStd_s2                           ,
        'yStd_s2':                            self.tree.yStd_s2                           ,
        'layerStd_s2':                        self.tree.layerStd_s2                       ,

        'energy_s3':                          self.tree.energy_s3                         ,
        'nHits_s3':                           self.tree.nHits_s3                          ,
        'xMean_s3':                           self.tree.xMean_s3                          ,
        'yMean_s3':                           self.tree.yMean_s3                          ,
        'layerMean_s3':                       self.tree.layerMean_s3                      ,
        'xStd_s3':                            self.tree.xStd_s3                           ,
        'yStd_s3':                            self.tree.yStd_s3                           ,
        'layerStd_s3':                        self.tree.layerStd_s3                       ,

        # Electron RoC variables
        'electronContainmentEnergy_x1_s1':    self.tree.electronContainmentEnergy_x1_s1   ,
        'electronContainmentEnergy_x2_s1':    self.tree.electronContainmentEnergy_x2_s1   ,
        'electronContainmentEnergy_x3_s1':    self.tree.electronContainmentEnergy_x3_s1   ,
        'electronContainmentEnergy_x4_s1':    self.tree.electronContainmentEnergy_x4_s1   ,
        'electronContainmentEnergy_x5_s1':    self.tree.electronContainmentEnergy_x5_s1   ,

        'electronContainmentNHits_x1_s1':     self.tree.electronContainmentNHits_x1_s1    ,
        'electronContainmentNHits_x2_s1':     self.tree.electronContainmentNHits_x2_s1    ,
        'electronContainmentNHits_x3_s1':     self.tree.electronContainmentNHits_x3_s1    ,
        'electronContainmentNHits_x4_s1':     self.tree.electronContainmentNHits_x4_s1    ,
        'electronContainmentNHits_x5_s1':     self.tree.electronContainmentNHits_x5_s1    ,

        'electronContainmentXMean_x1_s1':     self.tree.electronContainmentXMean_x1_s1    ,
        'electronContainmentXMean_x2_s1':     self.tree.electronContainmentXMean_x2_s1    ,
        'electronContainmentXMean_x3_s1':     self.tree.electronContainmentXMean_x3_s1    ,
        'electronContainmentXMean_x4_s1':     self.tree.electronContainmentXMean_x4_s1    ,
        'electronContainmentXMean_x5_s1':     self.tree.electronContainmentXMean_x5_s1    ,

        'electronContainmentYMean_x1_s1':     self.tree.electronContainmentYMean_x1_s1    ,
        'electronContainmentYMean_x2_s1':     self.tree.electronContainmentYMean_x2_s1    ,
        'electronContainmentYMean_x3_s1':     self.tree.electronContainmentYMean_x3_s1    ,
        'electronContainmentYMean_x4_s1':     self.tree.electronContainmentYMean_x4_s1    ,
        'electronContainmentYMean_x5_s1':     self.tree.electronContainmentYMean_x5_s1    ,

        'electronContainmentLayerMean_x1_s1': self.tree.electronContainmentLayerMean_x1_s1,
        'electronContainmentLayerMean_x2_s1': self.tree.electronContainmentLayerMean_x2_s1,
        'electronContainmentLayerMean_x3_s1': self.tree.electronContainmentLayerMean_x3_s1,
        'electronContainmentLayerMean_x4_s1': self.tree.electronContainmentLayerMean_x4_s1,
        'electronContainmentLayerMean_x5_s1': self.tree.electronContainmentLayerMean_x5_s1,

        'electronContainmentXStd_x1_s1':      self.tree.electronContainmentXStd_x1_s1     ,
        'electronContainmentXStd_x2_s1':      self.tree.electronContainmentXStd_x2_s1     ,
        'electronContainmentXStd_x3_s1':      self.tree.electronContainmentXStd_x3_s1     ,
        'electronContainmentXStd_x4_s1':      self.tree.electronContainmentXStd_x4_s1     ,
        'electronContainmentXStd_x5_s1':      self.tree.electronContainmentXStd_x5_s1     ,

        'electronContainmentYStd_x1_s1':      self.tree.electronContainmentYStd_x1_s1     ,
        'electronContainmentYStd_x2_s1':      self.tree.electronContainmentYStd_x2_s1     ,
        'electronContainmentYStd_x3_s1':      self.tree.electronContainmentYStd_x3_s1     ,
        'electronContainmentYStd_x4_s1':      self.tree.electronContainmentYStd_x4_s1     ,
        'electronContainmentYStd_x5_s1':      self.tree.electronContainmentYStd_x5_s1     ,

        'electronContainmentLayerStd_x1_s1':  self.tree.electronContainmentLayerStd_x1_s1 ,
        'electronContainmentLayerStd_x2_s1':  self.tree.electronContainmentLayerStd_x2_s1 ,
        'electronContainmentLayerStd_x3_s1':  self.tree.electronContainmentLayerStd_x3_s1 ,
        'electronContainmentLayerStd_x4_s1':  self.tree.electronContainmentLayerStd_x4_s1 ,
        'electronContainmentLayerStd_x5_s1':  self.tree.electronContainmentLayerStd_x5_s1 ,

        'electronContainmentEnergy_x1_s2':    self.tree.electronContainmentEnergy_x1_s2   ,
        'electronContainmentEnergy_x2_s2':    self.tree.electronContainmentEnergy_x2_s2   ,
        'electronContainmentEnergy_x3_s2':    self.tree.electronContainmentEnergy_x3_s2   ,
        'electronContainmentEnergy_x4_s2':    self.tree.electronContainmentEnergy_x4_s2   ,
        'electronContainmentEnergy_x5_s2':    self.tree.electronContainmentEnergy_x5_s2   ,

        'electronContainmentNHits_x1_s2':     self.tree.electronContainmentNHits_x1_s2    ,
        'electronContainmentNHits_x2_s2':     self.tree.electronContainmentNHits_x2_s2    ,
        'electronContainmentNHits_x3_s2':     self.tree.electronContainmentNHits_x3_s2    ,
        'electronContainmentNHits_x4_s2':     self.tree.electronContainmentNHits_x4_s2    ,
        'electronContainmentNHits_x5_s2':     self.tree.electronContainmentNHits_x5_s2    ,

        'electronContainmentXMean_x1_s2':     self.tree.electronContainmentXMean_x1_s2    ,
        'electronContainmentXMean_x2_s2':     self.tree.electronContainmentXMean_x2_s2    ,
        'electronContainmentXMean_x3_s2':     self.tree.electronContainmentXMean_x3_s2    ,
        'electronContainmentXMean_x4_s2':     self.tree.electronContainmentXMean_x4_s2    ,
        'electronContainmentXMean_x5_s2':     self.tree.electronContainmentXMean_x5_s2    ,

        'electronContainmentYMean_x1_s2':     self.tree.electronContainmentYMean_x1_s2    ,
        'electronContainmentYMean_x2_s2':     self.tree.electronContainmentYMean_x2_s2    ,
        'electronContainmentYMean_x3_s2':     self.tree.electronContainmentYMean_x3_s2    ,
        'electronContainmentYMean_x4_s2':     self.tree.electronContainmentYMean_x4_s2    ,
        'electronContainmentYMean_x5_s2':     self.tree.electronContainmentYMean_x5_s2    ,

        'electronContainmentLayerMean_x1_s2': self.tree.electronContainmentLayerMean_x1_s2,
        'electronContainmentLayerMean_x2_s2': self.tree.electronContainmentLayerMean_x2_s2,
        'electronContainmentLayerMean_x3_s2': self.tree.electronContainmentLayerMean_x3_s2,
        'electronContainmentLayerMean_x4_s2': self.tree.electronContainmentLayerMean_x4_s2,
        'electronContainmentLayerMean_x5_s2': self.tree.electronContainmentLayerMean_x5_s2,

        'electronContainmentXStd_x1_s2':      self.tree.electronContainmentXStd_x1_s2     ,
        'electronContainmentXStd_x2_s2':      self.tree.electronContainmentXStd_x2_s2     ,
        'electronContainmentXStd_x3_s2':      self.tree.electronContainmentXStd_x3_s2     ,
        'electronContainmentXStd_x4_s2':      self.tree.electronContainmentXStd_x4_s2     ,
        'electronContainmentXStd_x5_s2':      self.tree.electronContainmentXStd_x5_s2     ,

        'electronContainmentYStd_x1_s2':      self.tree.electronContainmentYStd_x1_s2     ,
        'electronContainmentYStd_x2_s2':      self.tree.electronContainmentYStd_x2_s2     ,
        'electronContainmentYStd_x3_s2':      self.tree.electronContainmentYStd_x3_s2     ,
        'electronContainmentYStd_x4_s2':      self.tree.electronContainmentYStd_x4_s2     ,
        'electronContainmentYStd_x5_s2':      self.tree.electronContainmentYStd_x5_s2     ,

        'electronContainmentLayerStd_x1_s2':  self.tree.electronContainmentLayerStd_x1_s2 ,
        'electronContainmentLayerStd_x2_s2':  self.tree.electronContainmentLayerStd_x2_s2 ,
        'electronContainmentLayerStd_x3_s2':  self.tree.electronContainmentLayerStd_x3_s2 ,
        'electronContainmentLayerStd_x4_s2':  self.tree.electronContainmentLayerStd_x4_s2 ,
        'electronContainmentLayerStd_x5_s2':  self.tree.electronContainmentLayerStd_x5_s2 ,

        'electronContainmentEnergy_x1_s3':    self.tree.electronContainmentEnergy_x1_s3   ,
        'electronContainmentEnergy_x2_s3':    self.tree.electronContainmentEnergy_x2_s3   ,
        'electronContainmentEnergy_x3_s3':    self.tree.electronContainmentEnergy_x3_s3   ,
        'electronContainmentEnergy_x4_s3':    self.tree.electronContainmentEnergy_x4_s3   ,
        'electronContainmentEnergy_x5_s3':    self.tree.electronContainmentEnergy_x5_s3   ,

        'electronContainmentNHits_x1_s3':     self.tree.electronContainmentNHits_x1_s3    ,
        'electronContainmentNHits_x2_s3':     self.tree.electronContainmentNHits_x2_s3    ,
        'electronContainmentNHits_x3_s3':     self.tree.electronContainmentNHits_x3_s3    ,
        'electronContainmentNHits_x4_s3':     self.tree.electronContainmentNHits_x4_s3    ,
        'electronContainmentNHits_x5_s3':     self.tree.electronContainmentNHits_x5_s3    ,

        'electronContainmentXMean_x1_s3':     self.tree.electronContainmentXMean_x1_s3    ,
        'electronContainmentXMean_x2_s3':     self.tree.electronContainmentXMean_x2_s3    ,
        'electronContainmentXMean_x3_s3':     self.tree.electronContainmentXMean_x3_s3    ,
        'electronContainmentXMean_x4_s3':     self.tree.electronContainmentXMean_x4_s3    ,
        'electronContainmentXMean_x5_s3':     self.tree.electronContainmentXMean_x5_s3    ,

        'electronContainmentYMean_x1_s3':     self.tree.electronContainmentYMean_x1_s3    ,
        'electronContainmentYMean_x2_s3':     self.tree.electronContainmentYMean_x2_s3    ,
        'electronContainmentYMean_x3_s3':     self.tree.electronContainmentYMean_x3_s3    ,
        'electronContainmentYMean_x4_s3':     self.tree.electronContainmentYMean_x4_s3    ,
        'electronContainmentYMean_x5_s3':     self.tree.electronContainmentYMean_x5_s3    ,

        'electronContainmentLayerMean_x1_s3': self.tree.electronContainmentLayerMean_x1_s3,
        'electronContainmentLayerMean_x2_s3': self.tree.electronContainmentLayerMean_x2_s3,
        'electronContainmentLayerMean_x3_s3': self.tree.electronContainmentLayerMean_x3_s3,
        'electronContainmentLayerMean_x4_s3': self.tree.electronContainmentLayerMean_x4_s3,
        'electronContainmentLayerMean_x5_s3': self.tree.electronContainmentLayerMean_x5_s3,

        'electronContainmentXStd_x1_s3':      self.tree.electronContainmentXStd_x1_s3     ,
        'electronContainmentXStd_x2_s3':      self.tree.electronContainmentXStd_x2_s3     ,
        'electronContainmentXStd_x3_s3':      self.tree.electronContainmentXStd_x3_s3     ,
        'electronContainmentXStd_x4_s3':      self.tree.electronContainmentXStd_x4_s3     ,
        'electronContainmentXStd_x5_s3':      self.tree.electronContainmentXStd_x5_s3     ,

        'electronContainmentYStd_x1_s3':      self.tree.electronContainmentYStd_x1_s3     ,
        'electronContainmentYStd_x2_s3':      self.tree.electronContainmentYStd_x2_s3     ,
        'electronContainmentYStd_x3_s3':      self.tree.electronContainmentYStd_x3_s3     ,
        'electronContainmentYStd_x4_s3':      self.tree.electronContainmentYStd_x4_s3     ,
        'electronContainmentYStd_x5_s3':      self.tree.electronContainmentYStd_x5_s3     ,

        'electronContainmentLayerStd_x1_s3':  self.tree.electronContainmentLayerStd_x1_s3 ,
        'electronContainmentLayerStd_x2_s3':  self.tree.electronContainmentLayerStd_x2_s3 ,
        'electronContainmentLayerStd_x3_s3':  self.tree.electronContainmentLayerStd_x3_s3 ,
        'electronContainmentLayerStd_x4_s3':  self.tree.electronContainmentLayerStd_x4_s3 ,
        'electronContainmentLayerStd_x5_s3':  self.tree.electronContainmentLayerStd_x5_s3 ,

        # Photon RoC variables
        'photonContainmentEnergy_x1_s1':      self.tree.photonContainmentEnergy_x1_s1     ,
        'photonContainmentEnergy_x2_s1':      self.tree.photonContainmentEnergy_x2_s1     ,
        'photonContainmentEnergy_x3_s1':      self.tree.photonContainmentEnergy_x3_s1     ,
        'photonContainmentEnergy_x4_s1':      self.tree.photonContainmentEnergy_x4_s1     ,
        'photonContainmentEnergy_x5_s1':      self.tree.photonContainmentEnergy_x5_s1     ,

        'photonContainmentNHits_x1_s1':       self.tree.photonContainmentNHits_x1_s1      ,
        'photonContainmentNHits_x2_s1':       self.tree.photonContainmentNHits_x2_s1      ,
        'photonContainmentNHits_x3_s1':       self.tree.photonContainmentNHits_x3_s1      ,
        'photonContainmentNHits_x4_s1':       self.tree.photonContainmentNHits_x4_s1      ,
        'photonContainmentNHits_x5_s1':       self.tree.photonContainmentNHits_x5_s1      ,

        'photonContainmentXMean_x1_s1':       self.tree.photonContainmentXMean_x1_s1      ,
        'photonContainmentXMean_x2_s1':       self.tree.photonContainmentXMean_x2_s1      ,
        'photonContainmentXMean_x3_s1':       self.tree.photonContainmentXMean_x3_s1      ,
        'photonContainmentXMean_x4_s1':       self.tree.photonContainmentXMean_x4_s1      ,
        'photonContainmentXMean_x5_s1':       self.tree.photonContainmentXMean_x5_s1      ,

        'photonContainmentYMean_x1_s1':       self.tree.photonContainmentYMean_x1_s1      ,
        'photonContainmentYMean_x2_s1':       self.tree.photonContainmentYMean_x2_s1      ,
        'photonContainmentYMean_x3_s1':       self.tree.photonContainmentYMean_x3_s1      ,
        'photonContainmentYMean_x4_s1':       self.tree.photonContainmentYMean_x4_s1      ,
        'photonContainmentYMean_x5_s1':       self.tree.photonContainmentYMean_x5_s1      ,

        'photonContainmentLayerMean_x1_s1':   self.tree.photonContainmentLayerMean_x1_s1  ,
        'photonContainmentLayerMean_x2_s1':   self.tree.photonContainmentLayerMean_x2_s1  ,
        'photonContainmentLayerMean_x3_s1':   self.tree.photonContainmentLayerMean_x3_s1  ,
        'photonContainmentLayerMean_x4_s1':   self.tree.photonContainmentLayerMean_x4_s1  ,
        'photonContainmentLayerMean_x5_s1':   self.tree.photonContainmentLayerMean_x5_s1  ,

        'photonContainmentXStd_x1_s1':        self.tree.photonContainmentXStd_x1_s1       ,
        'photonContainmentXStd_x2_s1':        self.tree.photonContainmentXStd_x2_s1       ,
        'photonContainmentXStd_x3_s1':        self.tree.photonContainmentXStd_x3_s1       ,
        'photonContainmentXStd_x4_s1':        self.tree.photonContainmentXStd_x4_s1       ,
        'photonContainmentXStd_x5_s1':        self.tree.photonContainmentXStd_x5_s1       ,

        'photonContainmentYStd_x1_s1':        self.tree.photonContainmentYStd_x1_s1       ,
        'photonContainmentYStd_x2_s1':        self.tree.photonContainmentYStd_x2_s1       ,
        'photonContainmentYStd_x3_s1':        self.tree.photonContainmentYStd_x3_s1       ,
        'photonContainmentYStd_x4_s1':        self.tree.photonContainmentYStd_x4_s1       ,
        'photonContainmentYStd_x5_s1':        self.tree.photonContainmentYStd_x5_s1       ,

        'photonContainmentLayerStd_x1_s1':    self.tree.photonContainmentLayerStd_x1_s1   ,
        'photonContainmentLayerStd_x2_s1':    self.tree.photonContainmentLayerStd_x2_s1   ,
        'photonContainmentLayerStd_x3_s1':    self.tree.photonContainmentLayerStd_x3_s1   ,
        'photonContainmentLayerStd_x4_s1':    self.tree.photonContainmentLayerStd_x4_s1   ,
        'photonContainmentLayerStd_x5_s1':    self.tree.photonContainmentLayerStd_x5_s1   ,

        'photonContainmentEnergy_x1_s2':      self.tree.photonContainmentEnergy_x1_s2     ,
        'photonContainmentEnergy_x2_s2':      self.tree.photonContainmentEnergy_x2_s2     ,
        'photonContainmentEnergy_x3_s2':      self.tree.photonContainmentEnergy_x3_s2     ,
        'photonContainmentEnergy_x4_s2':      self.tree.photonContainmentEnergy_x4_s2     ,
        'photonContainmentEnergy_x5_s2':      self.tree.photonContainmentEnergy_x5_s2     ,

        'photonContainmentNHits_x1_s2':       self.tree.photonContainmentNHits_x1_s2      ,
        'photonContainmentNHits_x2_s2':       self.tree.photonContainmentNHits_x2_s2      ,
        'photonContainmentNHits_x3_s2':       self.tree.photonContainmentNHits_x3_s2      ,
        'photonContainmentNHits_x4_s2':       self.tree.photonContainmentNHits_x4_s2      ,
        'photonContainmentNHits_x5_s2':       self.tree.photonContainmentNHits_x5_s2      ,

        'photonContainmentXMean_x1_s2':       self.tree.photonContainmentXMean_x1_s2      ,
        'photonContainmentXMean_x2_s2':       self.tree.photonContainmentXMean_x2_s2      ,
        'photonContainmentXMean_x3_s2':       self.tree.photonContainmentXMean_x3_s2      ,
        'photonContainmentXMean_x4_s2':       self.tree.photonContainmentXMean_x4_s2      ,
        'photonContainmentXMean_x5_s2':       self.tree.photonContainmentXMean_x5_s2      ,

        'photonContainmentYMean_x1_s2':       self.tree.photonContainmentYMean_x1_s2      ,
        'photonContainmentYMean_x2_s2':       self.tree.photonContainmentYMean_x2_s2      ,
        'photonContainmentYMean_x3_s2':       self.tree.photonContainmentYMean_x3_s2      ,
        'photonContainmentYMean_x4_s2':       self.tree.photonContainmentYMean_x4_s2      ,
        'photonContainmentYMean_x5_s2':       self.tree.photonContainmentYMean_x5_s2      ,

        'photonContainmentLayerMean_x1_s2':   self.tree.photonContainmentLayerMean_x1_s2  ,
        'photonContainmentLayerMean_x2_s2':   self.tree.photonContainmentLayerMean_x2_s2  ,
        'photonContainmentLayerMean_x3_s2':   self.tree.photonContainmentLayerMean_x3_s2  ,
        'photonContainmentLayerMean_x4_s2':   self.tree.photonContainmentLayerMean_x4_s2  ,
        'photonContainmentLayerMean_x5_s2':   self.tree.photonContainmentLayerMean_x5_s2  ,

        'photonContainmentXStd_x1_s2':        self.tree.photonContainmentXStd_x1_s2       ,
        'photonContainmentXStd_x2_s2':        self.tree.photonContainmentXStd_x2_s2       ,
        'photonContainmentXStd_x3_s2':        self.tree.photonContainmentXStd_x3_s2       ,
        'photonContainmentXStd_x4_s2':        self.tree.photonContainmentXStd_x4_s2       ,
        'photonContainmentXStd_x5_s2':        self.tree.photonContainmentXStd_x5_s2       ,

        'photonContainmentYStd_x1_s2':        self.tree.photonContainmentYStd_x1_s2       ,
        'photonContainmentYStd_x2_s2':        self.tree.photonContainmentYStd_x2_s2       ,
        'photonContainmentYStd_x3_s2':        self.tree.photonContainmentYStd_x3_s2       ,
        'photonContainmentYStd_x4_s2':        self.tree.photonContainmentYStd_x4_s2       ,
        'photonContainmentYStd_x5_s2':        self.tree.photonContainmentYStd_x5_s2       ,

        'photonContainmentLayerStd_x1_s2':    self.tree.photonContainmentLayerStd_x1_s2   ,
        'photonContainmentLayerStd_x2_s2':    self.tree.photonContainmentLayerStd_x2_s2   ,
        'photonContainmentLayerStd_x3_s2':    self.tree.photonContainmentLayerStd_x3_s2   ,
        'photonContainmentLayerStd_x4_s2':    self.tree.photonContainmentLayerStd_x4_s2   ,
        'photonContainmentLayerStd_x5_s2':    self.tree.photonContainmentLayerStd_x5_s2   ,

        'photonContainmentEnergy_x1_s3':      self.tree.photonContainmentEnergy_x1_s3     ,
        'photonContainmentEnergy_x2_s3':      self.tree.photonContainmentEnergy_x2_s3     ,
        'photonContainmentEnergy_x3_s3':      self.tree.photonContainmentEnergy_x3_s3     ,
        'photonContainmentEnergy_x4_s3':      self.tree.photonContainmentEnergy_x4_s3     ,
        'photonContainmentEnergy_x5_s3':      self.tree.photonContainmentEnergy_x5_s3     ,

        'photonContainmentNHits_x1_s3':       self.tree.photonContainmentNHits_x1_s3      ,
        'photonContainmentNHits_x2_s3':       self.tree.photonContainmentNHits_x2_s3      ,
        'photonContainmentNHits_x3_s3':       self.tree.photonContainmentNHits_x3_s3      ,
        'photonContainmentNHits_x4_s3':       self.tree.photonContainmentNHits_x4_s3      ,
        'photonContainmentNHits_x5_s3':       self.tree.photonContainmentNHits_x5_s3      ,

        'photonContainmentXMean_x1_s3':       self.tree.photonContainmentXMean_x1_s3      ,
        'photonContainmentXMean_x2_s3':       self.tree.photonContainmentXMean_x2_s3      ,
        'photonContainmentXMean_x3_s3':       self.tree.photonContainmentXMean_x3_s3      ,
        'photonContainmentXMean_x4_s3':       self.tree.photonContainmentXMean_x4_s3      ,
        'photonContainmentXMean_x5_s3':       self.tree.photonContainmentXMean_x5_s3      ,

        'photonContainmentYMean_x1_s3':       self.tree.photonContainmentYMean_x1_s3      ,
        'photonContainmentYMean_x2_s3':       self.tree.photonContainmentYMean_x2_s3      ,
        'photonContainmentYMean_x3_s3':       self.tree.photonContainmentYMean_x3_s3      ,
        'photonContainmentYMean_x4_s3':       self.tree.photonContainmentYMean_x4_s3      ,
        'photonContainmentYMean_x5_s3':       self.tree.photonContainmentYMean_x5_s3      ,

        'photonContainmentLayerMean_x1_s3':   self.tree.photonContainmentLayerMean_x1_s3  ,
        'photonContainmentLayerMean_x2_s3':   self.tree.photonContainmentLayerMean_x2_s3  ,
        'photonContainmentLayerMean_x3_s3':   self.tree.photonContainmentLayerMean_x3_s3  ,
        'photonContainmentLayerMean_x4_s3':   self.tree.photonContainmentLayerMean_x4_s3  ,
        'photonContainmentLayerMean_x5_s3':   self.tree.photonContainmentLayerMean_x5_s3  ,

        'photonContainmentXStd_x1_s3':        self.tree.photonContainmentXStd_x1_s3       ,
        'photonContainmentXStd_x2_s3':        self.tree.photonContainmentXStd_x2_s3       ,
        'photonContainmentXStd_x3_s3':        self.tree.photonContainmentXStd_x3_s3       ,
        'photonContainmentXStd_x4_s3':        self.tree.photonContainmentXStd_x4_s3       ,
        'photonContainmentXStd_x5_s3':        self.tree.photonContainmentXStd_x5_s3       ,

        'photonContainmentYStd_x1_s3':        self.tree.photonContainmentYStd_x1_s3       ,
        'photonContainmentYStd_x2_s3':        self.tree.photonContainmentYStd_x2_s3       ,
        'photonContainmentYStd_x3_s3':        self.tree.photonContainmentYStd_x3_s3       ,
        'photonContainmentYStd_x4_s3':        self.tree.photonContainmentYStd_x4_s3       ,
        'photonContainmentYStd_x5_s3':        self.tree.photonContainmentYStd_x5_s3       ,

        'photonContainmentLayerStd_x1_s3':    self.tree.photonContainmentLayerStd_x1_s3   ,
        'photonContainmentLayerStd_x2_s3':    self.tree.photonContainmentLayerStd_x2_s3   ,
        'photonContainmentLayerStd_x3_s3':    self.tree.photonContainmentLayerStd_x3_s3   ,
        'photonContainmentLayerStd_x4_s3':    self.tree.photonContainmentLayerStd_x4_s3   ,
        'photonContainmentLayerStd_x5_s3':    self.tree.photonContainmentLayerStd_x5_s3   ,

        # Outside RoC variables
        'outsideContainmentEnergy_x1_s1':     self.tree.outsideContainmentEnergy_x1_s1    ,
        'outsideContainmentEnergy_x2_s1':     self.tree.outsideContainmentEnergy_x2_s1    ,
        'outsideContainmentEnergy_x3_s1':     self.tree.outsideContainmentEnergy_x3_s1    ,
        'outsideContainmentEnergy_x4_s1':     self.tree.outsideContainmentEnergy_x4_s1    ,
        'outsideContainmentEnergy_x5_s1':     self.tree.outsideContainmentEnergy_x5_s1    ,

        'outsideContainmentNHits_x1_s1':      self.tree.outsideContainmentNHits_x1_s1     ,
        'outsideContainmentNHits_x2_s1':      self.tree.outsideContainmentNHits_x2_s1     ,
        'outsideContainmentNHits_x3_s1':      self.tree.outsideContainmentNHits_x3_s1     ,
        'outsideContainmentNHits_x4_s1':      self.tree.outsideContainmentNHits_x4_s1     ,
        'outsideContainmentNHits_x5_s1':      self.tree.outsideContainmentNHits_x5_s1     ,

        'outsideContainmentXMean_x1_s1':      self.tree.outsideContainmentXMean_x1_s1     ,
        'outsideContainmentXMean_x2_s1':      self.tree.outsideContainmentXMean_x2_s1     ,
        'outsideContainmentXMean_x3_s1':      self.tree.outsideContainmentXMean_x3_s1     ,
        'outsideContainmentXMean_x4_s1':      self.tree.outsideContainmentXMean_x4_s1     ,
        'outsideContainmentXMean_x5_s1':      self.tree.outsideContainmentXMean_x5_s1     ,

        'outsideContainmentYMean_x1_s1':      self.tree.outsideContainmentYMean_x1_s1     ,
        'outsideContainmentYMean_x2_s1':      self.tree.outsideContainmentYMean_x2_s1     ,
        'outsideContainmentYMean_x3_s1':      self.tree.outsideContainmentYMean_x3_s1     ,
        'outsideContainmentYMean_x4_s1':      self.tree.outsideContainmentYMean_x4_s1     ,
        'outsideContainmentYMean_x5_s1':      self.tree.outsideContainmentYMean_x5_s1     ,

        'outsideContainmentLayerMean_x1_s1':  self.tree.outsideContainmentLayerMean_x1_s1 ,
        'outsideContainmentLayerMean_x2_s1':  self.tree.outsideContainmentLayerMean_x2_s1 ,
        'outsideContainmentLayerMean_x3_s1':  self.tree.outsideContainmentLayerMean_x3_s1 ,
        'outsideContainmentLayerMean_x4_s1':  self.tree.outsideContainmentLayerMean_x4_s1 ,
        'outsideContainmentLayerMean_x5_s1':  self.tree.outsideContainmentLayerMean_x5_s1 ,

        'outsideContainmentXStd_x1_s1':       self.tree.outsideContainmentXStd_x1_s1      ,
        'outsideContainmentXStd_x2_s1':       self.tree.outsideContainmentXStd_x2_s1      ,
        'outsideContainmentXStd_x3_s1':       self.tree.outsideContainmentXStd_x3_s1      ,
        'outsideContainmentXStd_x4_s1':       self.tree.outsideContainmentXStd_x4_s1      ,
        'outsideContainmentXStd_x5_s1':       self.tree.outsideContainmentXStd_x5_s1      ,

        'outsideContainmentYStd_x1_s1':       self.tree.outsideContainmentYStd_x1_s1      ,
        'outsideContainmentYStd_x2_s1':       self.tree.outsideContainmentYStd_x2_s1      ,
        'outsideContainmentYStd_x3_s1':       self.tree.outsideContainmentYStd_x3_s1      ,
        'outsideContainmentYStd_x4_s1':       self.tree.outsideContainmentYStd_x4_s1      ,
        'outsideContainmentYStd_x5_s1':       self.tree.outsideContainmentYStd_x5_s1      ,

        'outsideContainmentLayerStd_x1_s1':   self.tree.outsideContainmentLayerStd_x1_s1  ,
        'outsideContainmentLayerStd_x2_s1':   self.tree.outsideContainmentLayerStd_x2_s1  ,
        'outsideContainmentLayerStd_x3_s1':   self.tree.outsideContainmentLayerStd_x3_s1  ,
        'outsideContainmentLayerStd_x4_s1':   self.tree.outsideContainmentLayerStd_x4_s1  ,
        'outsideContainmentLayerStd_x5_s1':   self.tree.outsideContainmentLayerStd_x5_s1  ,

        'outsideContainmentEnergy_x1_s2':     self.tree.outsideContainmentEnergy_x1_s2    ,
        'outsideContainmentEnergy_x2_s2':     self.tree.outsideContainmentEnergy_x2_s2    ,
        'outsideContainmentEnergy_x3_s2':     self.tree.outsideContainmentEnergy_x3_s2    ,
        'outsideContainmentEnergy_x4_s2':     self.tree.outsideContainmentEnergy_x4_s2    ,
        'outsideContainmentEnergy_x5_s2':     self.tree.outsideContainmentEnergy_x5_s2    ,

        'outsideContainmentNHits_x1_s2':      self.tree.outsideContainmentNHits_x1_s2     ,
        'outsideContainmentNHits_x2_s2':      self.tree.outsideContainmentNHits_x2_s2     ,
        'outsideContainmentNHits_x3_s2':      self.tree.outsideContainmentNHits_x3_s2     ,
        'outsideContainmentNHits_x4_s2':      self.tree.outsideContainmentNHits_x4_s2     ,
        'outsideContainmentNHits_x5_s2':      self.tree.outsideContainmentNHits_x5_s2     ,

        'outsideContainmentXMean_x1_s2':      self.tree.outsideContainmentXMean_x1_s2     ,
        'outsideContainmentXMean_x2_s2':      self.tree.outsideContainmentXMean_x2_s2     ,
        'outsideContainmentXMean_x3_s2':      self.tree.outsideContainmentXMean_x3_s2     ,
        'outsideContainmentXMean_x4_s2':      self.tree.outsideContainmentXMean_x4_s2     ,
        'outsideContainmentXMean_x5_s2':      self.tree.outsideContainmentXMean_x5_s2     ,

        'outsideContainmentYMean_x1_s2':      self.tree.outsideContainmentYMean_x1_s2     ,
        'outsideContainmentYMean_x2_s2':      self.tree.outsideContainmentYMean_x2_s2     ,
        'outsideContainmentYMean_x3_s2':      self.tree.outsideContainmentYMean_x3_s2     ,
        'outsideContainmentYMean_x4_s2':      self.tree.outsideContainmentYMean_x4_s2     ,
        'outsideContainmentYMean_x5_s2':      self.tree.outsideContainmentYMean_x5_s2     ,

        'outsideContainmentLayerMean_x1_s2':  self.tree.outsideContainmentLayerMean_x1_s2 ,
        'outsideContainmentLayerMean_x2_s2':  self.tree.outsideContainmentLayerMean_x2_s2 ,
        'outsideContainmentLayerMean_x3_s2':  self.tree.outsideContainmentLayerMean_x3_s2 ,
        'outsideContainmentLayerMean_x4_s2':  self.tree.outsideContainmentLayerMean_x4_s2 ,
        'outsideContainmentLayerMean_x5_s2':  self.tree.outsideContainmentLayerMean_x5_s2 ,

        'outsideContainmentXStd_x1_s2':       self.tree.outsideContainmentXStd_x1_s2      ,
        'outsideContainmentXStd_x2_s2':       self.tree.outsideContainmentXStd_x2_s2      ,
        'outsideContainmentXStd_x3_s2':       self.tree.outsideContainmentXStd_x3_s2      ,
        'outsideContainmentXStd_x4_s2':       self.tree.outsideContainmentXStd_x4_s2      ,
        'outsideContainmentXStd_x5_s2':       self.tree.outsideContainmentXStd_x5_s2      ,

        'outsideContainmentYStd_x1_s2':       self.tree.outsideContainmentYStd_x1_s2      ,
        'outsideContainmentYStd_x2_s2':       self.tree.outsideContainmentYStd_x2_s2      ,
        'outsideContainmentYStd_x3_s2':       self.tree.outsideContainmentYStd_x3_s2      ,
        'outsideContainmentYStd_x4_s2':       self.tree.outsideContainmentYStd_x4_s2      ,
        'outsideContainmentYStd_x5_s2':       self.tree.outsideContainmentYStd_x5_s2      ,

        'outsideContainmentLayerStd_x1_s2':   self.tree.outsideContainmentLayerStd_x1_s2  ,
        'outsideContainmentLayerStd_x2_s2':   self.tree.outsideContainmentLayerStd_x2_s2  ,
        'outsideContainmentLayerStd_x3_s2':   self.tree.outsideContainmentLayerStd_x3_s2  ,
        'outsideContainmentLayerStd_x4_s2':   self.tree.outsideContainmentLayerStd_x4_s2  ,
        'outsideContainmentLayerStd_x5_s2':   self.tree.outsideContainmentLayerStd_x5_s2  ,

        'outsideContainmentEnergy_x1_s3':     self.tree.outsideContainmentEnergy_x1_s3    ,
        'outsideContainmentEnergy_x2_s3':     self.tree.outsideContainmentEnergy_x2_s3    ,
        'outsideContainmentEnergy_x3_s3':     self.tree.outsideContainmentEnergy_x3_s3    ,
        'outsideContainmentEnergy_x4_s3':     self.tree.outsideContainmentEnergy_x4_s3    ,
        'outsideContainmentEnergy_x5_s3':     self.tree.outsideContainmentEnergy_x5_s3    ,

        'outsideContainmentNHits_x1_s3':      self.tree.outsideContainmentNHits_x1_s3     ,
        'outsideContainmentNHits_x2_s3':      self.tree.outsideContainmentNHits_x2_s3     ,
        'outsideContainmentNHits_x3_s3':      self.tree.outsideContainmentNHits_x3_s3     ,
        'outsideContainmentNHits_x4_s3':      self.tree.outsideContainmentNHits_x4_s3     ,
        'outsideContainmentNHits_x5_s3':      self.tree.outsideContainmentNHits_x5_s3     ,

        'outsideContainmentXMean_x1_s3':      self.tree.outsideContainmentXMean_x1_s3     ,
        'outsideContainmentXMean_x2_s3':      self.tree.outsideContainmentXMean_x2_s3     ,
        'outsideContainmentXMean_x3_s3':      self.tree.outsideContainmentXMean_x3_s3     ,
        'outsideContainmentXMean_x4_s3':      self.tree.outsideContainmentXMean_x4_s3     ,
        'outsideContainmentXMean_x5_s3':      self.tree.outsideContainmentXMean_x5_s3     ,

        'outsideContainmentYMean_x1_s3':      self.tree.outsideContainmentYMean_x1_s3     ,
        'outsideContainmentYMean_x2_s3':      self.tree.outsideContainmentYMean_x2_s3     ,
        'outsideContainmentYMean_x3_s3':      self.tree.outsideContainmentYMean_x3_s3     ,
        'outsideContainmentYMean_x4_s3':      self.tree.outsideContainmentYMean_x4_s3     ,
        'outsideContainmentYMean_x5_s3':      self.tree.outsideContainmentYMean_x5_s3     ,

        'outsideContainmentLayerMean_x1_s3':  self.tree.outsideContainmentLayerMean_x1_s3 ,
        'outsideContainmentLayerMean_x2_s3':  self.tree.outsideContainmentLayerMean_x2_s3 ,
        'outsideContainmentLayerMean_x3_s3':  self.tree.outsideContainmentLayerMean_x3_s3 ,
        'outsideContainmentLayerMean_x4_s3':  self.tree.outsideContainmentLayerMean_x4_s3 ,
        'outsideContainmentLayerMean_x5_s3':  self.tree.outsideContainmentLayerMean_x5_s3 ,

        'outsideContainmentXStd_x1_s3':       self.tree.outsideContainmentXStd_x1_s3      ,
        'outsideContainmentXStd_x2_s3':       self.tree.outsideContainmentXStd_x2_s3      ,
        'outsideContainmentXStd_x3_s3':       self.tree.outsideContainmentXStd_x3_s3      ,
        'outsideContainmentXStd_x4_s3':       self.tree.outsideContainmentXStd_x4_s3      ,
        'outsideContainmentXStd_x5_s3':       self.tree.outsideContainmentXStd_x5_s3      ,

        'outsideContainmentYStd_x1_s3':       self.tree.outsideContainmentYStd_x1_s3      ,
        'outsideContainmentYStd_x2_s3':       self.tree.outsideContainmentYStd_x2_s3      ,
        'outsideContainmentYStd_x3_s3':       self.tree.outsideContainmentYStd_x3_s3      ,
        'outsideContainmentYStd_x4_s3':       self.tree.outsideContainmentYStd_x4_s3      ,
        'outsideContainmentYStd_x5_s3':       self.tree.outsideContainmentYStd_x5_s3      ,

        'outsideContainmentLayerStd_x1_s3':   self.tree.outsideContainmentLayerStd_x1_s3  ,
        'outsideContainmentLayerStd_x2_s3':   self.tree.outsideContainmentLayerStd_x2_s3  ,
        'outsideContainmentLayerStd_x3_s3':   self.tree.outsideContainmentLayerStd_x3_s3  ,
        'outsideContainmentLayerStd_x4_s3':   self.tree.outsideContainmentLayerStd_x4_s3  ,
        'outsideContainmentLayerStd_x5_s3':   self.tree.outsideContainmentLayerStd_x5_s3
    }

    # Add the prediction to the dictionary
    evt = [new_values[branch_name] for branch_name in new_values]
    events = np.array([evt])
    pred = float(model.predict(xgb.DMatrix(events))[0])
    new_values['discValue'] = pred

    # Fill the tree with new values
    self.tree_model.fill(new_values)


###############
# RUN
###############

if __name__ == '__main__':
    main()
