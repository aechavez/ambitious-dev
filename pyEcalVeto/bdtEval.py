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
    'nReadoutHits':              {'dtype': int,   'default': 0 },
    'summedDet':                 {'dtype': float, 'default': 0.},
    'summedTightIso':            {'dtype': float, 'default': 0.},
    'maxCellDep':                {'dtype': float, 'default': 0.},
    'showerRMS':                 {'dtype': float, 'default': 0.},
    'xStd':                      {'dtype': float, 'default': 0.},
    'yStd':                      {'dtype': float, 'default': 0.},
    'avgLayerHit':               {'dtype': float, 'default': 0.},
    'stdLayerHit':               {'dtype': float, 'default': 0.},
    'deepestLayerHit':           {'dtype': int,   'default': 0 },
    'ecalBackEnergy':            {'dtype': float, 'default': 0.},

    # MIP tracking variables
    'straight4':                 {'dtype': int,   'default': 0 },
    'firstNearPhotonLayer':      {'dtype': int,   'default': 33},
    'nNearPhotonHits':           {'dtype': int,   'default': 0 },
    'fullElectronTerritoryHits': {'dtype': int,   'default': 0 },
    'fullPhotonTerritoryHits':   {'dtype': int,   'default': 0 },
    'fullTerritoryRatio':        {'dtype': float, 'default': 1.},
    'electronTerritoryHits':     {'dtype': int,   'default': 0 },
    'photonTerritoryHits':       {'dtype': int,   'default': 0 },
    'territoryRatio':            {'dtype': float, 'default': 1.},
    'trajectorySeparation':      {'dtype': float, 'default': 0.},
    'trajectoryDot':             {'dtype': float, 'default': 0.}
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
        branch_information['photonContainmentEnergy_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentNHits_x{}_s{}'.format(j, i)]     = {'dtype': int,   'default': 0 }
        branch_information['photonContainmentXMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentYMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentLayerMean_x{}_s{}'.format(j, i)] = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentXStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentYStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['photonContainmentLayerStd_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}

        # Outside RoC variables
        branch_information['outsideContainmentEnergy_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentNHits_x{}_s{}'.format(j, i)]     = {'dtype': int,   'default': 0 }
        branch_information['outsideContainmentXMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentYMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentLayerMean_x{}_s{}'.format(j, i)] = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentXStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentYStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['outsideContainmentLayerStd_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}

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

def event_process(self):

    # Feature list from input tree
    # Exp: feats = [ feat_value for feat_value in self.tree~ ]
    # Put all segmentation variables in for now (Take out the ones we won't need once
    # we make sure that all the python bdt stuff works)
    feats = [
            # Base variables
            self.tree.nReadoutHits              ,
            self.tree.summedDet                 ,
            self.tree.summedTightIso            ,
            self.tree.maxCellDep                ,
            self.tree.showerRMS                 ,
            self.tree.xStd                      ,
            self.tree.yStd                      ,
            self.tree.avgLayerHit               ,
            self.tree.stdLayerHit               ,
            self.tree.deepestLayerHit           ,
            self.tree.ecalBackEnergy            ,
            # MIP Tracking variables
            self.tree.straight4                 ,
            self.tree.firstNearPhLayer          ,
            self.tree.nNearPhHits               ,
            self.tree.fullElectronTerritoryHits ,
            self.tree.fullPhotonTerritoryHits   ,
            self.tree.fullTerritoryRatio        ,
            self.tree.electronTerritoryHits     ,
            self.tree.photonTerritoryHits       ,
            self.tree.TerritoryRatio            ,
            # Longitudinal segment variables
            self.tree.energy_s1                 ,
            self.tree.nHits_s1                  ,
            self.tree.xMean_s1                  ,
            self.tree.yMean_s1                  ,
            self.tree.layerMean_s1              ,
            self.tree.xStd_s1                   ,
            self.tree.yStd_s1                   ,
            self.tree.layerStd_s1               ,
            self.tree.energy_s2                 ,
            self.tree.nHits_s2                  ,
            self.tree.xMean_s2                  ,
            self.tree.yMean_s2                  ,
            self.tree.layerMean_s2              ,
            self.tree.xStd_s2                   ,
            self.tree.yStd_s2                   ,
            self.tree.layerStd_s2               ,
            self.tree.energy_s3                 ,
            self.tree.nHits_s3                  ,
            self.tree.xMean_s3                  ,
            self.tree.yMean_s3                  ,
            self.tree.layerMean_s3              ,
            self.tree.xStd_s3                   ,
            self.tree.yStd_s3                   ,
            self.tree.layerStd_s3               ,
            # Electron RoC variables
            self.tree.eContEnergy_x1_s1         ,
            self.tree.eContEnergy_x2_s1         ,
            self.tree.eContEnergy_x3_s1         ,
            self.tree.eContEnergy_x4_s1         ,
            self.tree.eContEnergy_x5_s1         ,
            self.tree.eContNHits_x1_s1          ,
            self.tree.eContNHits_x2_s1          ,
            self.tree.eContNHits_x3_s1          ,
            self.tree.eContNHits_x4_s1          ,
            self.tree.eContNHits_x5_s1          ,
            self.tree.eContXMean_x1_s1          ,
            self.tree.eContXMean_x2_s1          ,
            self.tree.eContXMean_x3_s1          ,
            self.tree.eContXMean_x4_s1          ,
            self.tree.eContXMean_x5_s1          ,
            self.tree.eContYMean_x1_s1          ,
            self.tree.eContYMean_x2_s1          ,
            self.tree.eContYMean_x3_s1          ,
            self.tree.eContYMean_x4_s1          ,
            self.tree.eContYMean_x5_s1          ,
            self.tree.eContLayerMean_x1_s1      ,
            self.tree.eContLayerMean_x2_s1      ,
            self.tree.eContLayerMean_x3_s1      ,
            self.tree.eContLayerMean_x4_s1      ,
            self.tree.eContLayerMean_x5_s1      ,
            self.tree.eContXStd_x1_s1           ,
            self.tree.eContXStd_x2_s1           ,
            self.tree.eContXStd_x3_s1           ,
            self.tree.eContXStd_x4_s1           ,
            self.tree.eContXStd_x5_s1           ,
            self.tree.eContYStd_x1_s1           ,
            self.tree.eContYStd_x2_s1           ,
            self.tree.eContYStd_x3_s1           ,
            self.tree.eContYStd_x4_s1           ,
            self.tree.eContYStd_x5_s1           ,
            self.tree.eContLayerStd_x1_s1       ,
            self.tree.eContLayerStd_x2_s1       ,
            self.tree.eContLayerStd_x3_s1       ,
            self.tree.eContLayerStd_x4_s1       ,
            self.tree.eContLayerStd_x5_s1       ,
            self.tree.eContEnergy_x1_s2         ,
            self.tree.eContEnergy_x2_s2         ,
            self.tree.eContEnergy_x3_s2         ,
            self.tree.eContEnergy_x4_s2         ,
            self.tree.eContEnergy_x5_s2         ,
            self.tree.eContNHits_x1_s2          ,
            self.tree.eContNHits_x2_s2          ,
            self.tree.eContNHits_x3_s2          ,
            self.tree.eContNHits_x4_s2          ,
            self.tree.eContNHits_x5_s2          ,
            self.tree.eContXMean_x1_s2          ,
            self.tree.eContXMean_x2_s2          ,
            self.tree.eContXMean_x3_s2          ,
            self.tree.eContXMean_x4_s2          ,
            self.tree.eContXMean_x5_s2          ,
            self.tree.eContYMean_x1_s2          ,
            self.tree.eContYMean_x2_s2          ,
            self.tree.eContYMean_x3_s2          ,
            self.tree.eContYMean_x4_s2          ,
            self.tree.eContYMean_x5_s2          ,
            self.tree.eContLayerMean_x1_s2      ,
            self.tree.eContLayerMean_x2_s2      ,
            self.tree.eContLayerMean_x3_s2      ,
            self.tree.eContLayerMean_x4_s2      ,
            self.tree.eContLayerMean_x5_s2      ,
            self.tree.eContXStd_x1_s2           ,
            self.tree.eContXStd_x2_s2           ,
            self.tree.eContXStd_x3_s2           ,
            self.tree.eContXStd_x4_s2           ,
            self.tree.eContXStd_x5_s2           ,
            self.tree.eContYStd_x1_s2           ,
            self.tree.eContYStd_x2_s2           ,
            self.tree.eContYStd_x3_s2           ,
            self.tree.eContYStd_x4_s2           ,
            self.tree.eContYStd_x5_s2           ,
            self.tree.eContLayerStd_x1_s2       ,
            self.tree.eContLayerStd_x2_s2       ,
            self.tree.eContLayerStd_x3_s2       ,
            self.tree.eContLayerStd_x4_s2       ,
            self.tree.eContLayerStd_x5_s2       ,
            self.tree.eContEnergy_x1_s3         ,
            self.tree.eContEnergy_x2_s3         ,
            self.tree.eContEnergy_x3_s3         ,
            self.tree.eContEnergy_x4_s3         ,
            self.tree.eContEnergy_x5_s3         ,
            self.tree.eContNHits_x1_s3          ,
            self.tree.eContNHits_x2_s3          ,
            self.tree.eContNHits_x3_s3          ,
            self.tree.eContNHits_x4_s3          ,
            self.tree.eContNHits_x5_s3          ,
            self.tree.eContXMean_x1_s3          ,
            self.tree.eContXMean_x2_s3          ,
            self.tree.eContXMean_x3_s3          ,
            self.tree.eContXMean_x4_s3          ,
            self.tree.eContXMean_x5_s3          ,
            self.tree.eContYMean_x1_s3          ,
            self.tree.eContYMean_x2_s3          ,
            self.tree.eContYMean_x3_s3          ,
            self.tree.eContYMean_x4_s3          ,
            self.tree.eContYMean_x5_s3          ,
            self.tree.eContLayerMean_x1_s3      ,
            self.tree.eContLayerMean_x2_s3      ,
            self.tree.eContLayerMean_x3_s3      ,
            self.tree.eContLayerMean_x4_s3      ,
            self.tree.eContLayerMean_x5_s3      ,
            self.tree.eContXStd_x1_s3           ,
            self.tree.eContXStd_x2_s3           ,
            self.tree.eContXStd_x3_s3           ,
            self.tree.eContXStd_x4_s3           ,
            self.tree.eContXStd_x5_s3           ,
            self.tree.eContYStd_x1_s3           ,
            self.tree.eContYStd_x2_s3           ,
            self.tree.eContYStd_x3_s3           ,
            self.tree.eContYStd_x4_s3           ,
            self.tree.eContYStd_x5_s3           ,
            self.tree.eContLayerStd_x1_s3       ,
            self.tree.eContLayerStd_x2_s3       ,
            self.tree.eContLayerStd_x3_s3       ,
            self.tree.eContLayerStd_x4_s3       ,
            self.tree.eContLayerStd_x5_s3       ,
            # Photon RoC variables
            self.tree.gContEnergy_x1_s1         ,
            self.tree.gContEnergy_x2_s1         ,
            self.tree.gContEnergy_x3_s1         ,
            self.tree.gContEnergy_x4_s1         ,
            self.tree.gContEnergy_x5_s1         ,
            self.tree.gContNHits_x1_s1          ,
            self.tree.gContNHits_x2_s1          ,
            self.tree.gContNHits_x3_s1          ,
            self.tree.gContNHits_x4_s1          ,
            self.tree.gContNHits_x5_s1          ,
            self.tree.gContXMean_x1_s1          ,
            self.tree.gContXMean_x2_s1          ,
            self.tree.gContXMean_x3_s1          ,
            self.tree.gContXMean_x4_s1          ,
            self.tree.gContXMean_x5_s1          ,
            self.tree.gContYMean_x1_s1          ,
            self.tree.gContYMean_x2_s1          ,
            self.tree.gContYMean_x3_s1          ,
            self.tree.gContYMean_x4_s1          ,
            self.tree.gContYMean_x5_s1          ,
            self.tree.gContLayerMean_x1_s1      ,
            self.tree.gContLayerMean_x2_s1      ,
            self.tree.gContLayerMean_x3_s1      ,
            self.tree.gContLayerMean_x4_s1      ,
            self.tree.gContLayerMean_x5_s1      ,
            self.tree.gContXStd_x1_s1           ,
            self.tree.gContXStd_x2_s1           ,
            self.tree.gContXStd_x3_s1           ,
            self.tree.gContXStd_x4_s1           ,
            self.tree.gContXStd_x5_s1           ,
            self.tree.gContYStd_x1_s1           ,
            self.tree.gContYStd_x2_s1           ,
            self.tree.gContYStd_x3_s1           ,
            self.tree.gContYStd_x4_s1           ,
            self.tree.gContYStd_x5_s1           ,
            self.tree.gContLayerStd_x1_s1       ,
            self.tree.gContLayerStd_x2_s1       ,
            self.tree.gContLayerStd_x3_s1       ,
            self.tree.gContLayerStd_x4_s1       ,
            self.tree.gContLayerStd_x5_s1       ,
            self.tree.gContEnergy_x1_s2         ,
            self.tree.gContEnergy_x2_s2         ,
            self.tree.gContEnergy_x3_s2         ,
            self.tree.gContEnergy_x4_s2         ,
            self.tree.gContEnergy_x5_s2         ,
            self.tree.gContNHits_x1_s2          ,
            self.tree.gContNHits_x2_s2          ,
            self.tree.gContNHits_x3_s2          ,
            self.tree.gContNHits_x4_s2          ,
            self.tree.gContNHits_x5_s2          ,
            self.tree.gContXMean_x1_s2          ,
            self.tree.gContXMean_x2_s2          ,
            self.tree.gContXMean_x3_s2          ,
            self.tree.gContXMean_x4_s2          ,
            self.tree.gContXMean_x5_s2          ,
            self.tree.gContYMean_x1_s2          ,
            self.tree.gContYMean_x2_s2          ,
            self.tree.gContYMean_x3_s2          ,
            self.tree.gContYMean_x4_s2          ,
            self.tree.gContYMean_x5_s2          ,
            self.tree.gContLayerMean_x1_s2      ,
            self.tree.gContLayerMean_x2_s2      ,
            self.tree.gContLayerMean_x3_s2      ,
            self.tree.gContLayerMean_x4_s2      ,
            self.tree.gContLayerMean_x5_s2      ,
            self.tree.gContXStd_x1_s2           ,
            self.tree.gContXStd_x2_s2           ,
            self.tree.gContXStd_x3_s2           ,
            self.tree.gContXStd_x4_s2           ,
            self.tree.gContXStd_x5_s2           ,
            self.tree.gContYStd_x1_s2           ,
            self.tree.gContYStd_x2_s2           ,
            self.tree.gContYStd_x3_s2           ,
            self.tree.gContYStd_x4_s2           ,
            self.tree.gContYStd_x5_s2           ,
            self.tree.gContLayerStd_x1_s2       ,
            self.tree.gContLayerStd_x2_s2       ,
            self.tree.gContLayerStd_x3_s2       ,
            self.tree.gContLayerStd_x4_s2       ,
            self.tree.gContLayerStd_x5_s2       ,
            self.tree.gContEnergy_x1_s3         ,
            self.tree.gContEnergy_x2_s3         ,
            self.tree.gContEnergy_x3_s3         ,
            self.tree.gContEnergy_x4_s3         ,
            self.tree.gContEnergy_x5_s3         ,
            self.tree.gContNHits_x1_s3          ,
            self.tree.gContNHits_x2_s3          ,
            self.tree.gContNHits_x3_s3          ,
            self.tree.gContNHits_x4_s3          ,
            self.tree.gContNHits_x5_s3          ,
            self.tree.gContXMean_x1_s3          ,
            self.tree.gContXMean_x2_s3          ,
            self.tree.gContXMean_x3_s3          ,
            self.tree.gContXMean_x4_s3          ,
            self.tree.gContXMean_x5_s3          ,
            self.tree.gContYMean_x1_s3          ,
            self.tree.gContYMean_x2_s3          ,
            self.tree.gContYMean_x3_s3          ,
            self.tree.gContYMean_x4_s3          ,
            self.tree.gContYMean_x5_s3          ,
            self.tree.gContLayerMean_x1_s3      ,
            self.tree.gContLayerMean_x2_s3      ,
            self.tree.gContLayerMean_x3_s3      ,
            self.tree.gContLayerMean_x4_s3      ,
            self.tree.gContLayerMean_x5_s3      ,
            self.tree.gContXStd_x1_s3           ,
            self.tree.gContXStd_x2_s3           ,
            self.tree.gContXStd_x3_s3           ,
            self.tree.gContXStd_x4_s3           ,
            self.tree.gContXStd_x5_s3           ,
            self.tree.gContYStd_x1_s3           ,
            self.tree.gContYStd_x2_s3           ,
            self.tree.gContYStd_x3_s3           ,
            self.tree.gContYStd_x4_s3           ,
            self.tree.gContYStd_x5_s3           ,
            self.tree.gContLayerStd_x1_s3       ,
            self.tree.gContLayerStd_x2_s3       ,
            self.tree.gContLayerStd_x3_s3       ,
            self.tree.gContLayerStd_x4_s3       ,
            self.tree.gContLayerStd_x5_s3       ,
            # Outside RoC variables
            self.tree.oContEnergy_x1_s1         ,
            self.tree.oContEnergy_x2_s1         ,
            self.tree.oContEnergy_x3_s1         ,
            self.tree.oContEnergy_x4_s1         ,
            self.tree.oContEnergy_x5_s1         ,
            self.tree.oContNHits_x1_s1          ,
            self.tree.oContNHits_x2_s1          ,
            self.tree.oContNHits_x3_s1          ,
            self.tree.oContNHits_x4_s1          ,
            self.tree.oContNHits_x5_s1          ,
            self.tree.oContXMean_x1_s1          ,
            self.tree.oContXMean_x2_s1          ,
            self.tree.oContXMean_x3_s1          ,
            self.tree.oContXMean_x4_s1          ,
            self.tree.oContXMean_x5_s1          ,
            self.tree.oContYMean_x1_s1          ,
            self.tree.oContYMean_x2_s1          ,
            self.tree.oContYMean_x3_s1          ,
            self.tree.oContYMean_x4_s1          ,
            self.tree.oContYMean_x5_s1          ,
            self.tree.oContLayerMean_x1_s1      ,
            self.tree.oContLayerMean_x2_s1      ,
            self.tree.oContLayerMean_x3_s1      ,
            self.tree.oContLayerMean_x4_s1      ,
            self.tree.oContLayerMean_x5_s1      ,
            self.tree.oContXStd_x1_s1           ,
            self.tree.oContXStd_x2_s1           ,
            self.tree.oContXStd_x3_s1           ,
            self.tree.oContXStd_x4_s1           ,
            self.tree.oContXStd_x5_s1           ,
            self.tree.oContYStd_x1_s1           ,
            self.tree.oContYStd_x2_s1           ,
            self.tree.oContYStd_x3_s1           ,
            self.tree.oContYStd_x4_s1           ,
            self.tree.oContYStd_x5_s1           ,
            self.tree.oContLayerStd_x1_s1       ,
            self.tree.oContLayerStd_x2_s1       ,
            self.tree.oContLayerStd_x3_s1       ,
            self.tree.oContLayerStd_x4_s1       ,
            self.tree.oContLayerStd_x5_s1       ,
            self.tree.oContEnergy_x1_s2         ,
            self.tree.oContEnergy_x2_s2         ,
            self.tree.oContEnergy_x3_s2         ,
            self.tree.oContEnergy_x4_s2         ,
            self.tree.oContEnergy_x5_s2         ,
            self.tree.oContNHits_x1_s2          ,
            self.tree.oContNHits_x2_s2          ,
            self.tree.oContNHits_x3_s2          ,
            self.tree.oContNHits_x4_s2          ,
            self.tree.oContNHits_x5_s2          ,
            self.tree.oContXMean_x1_s2          ,
            self.tree.oContXMean_x2_s2          ,
            self.tree.oContXMean_x3_s2          ,
            self.tree.oContXMean_x4_s2          ,
            self.tree.oContXMean_x5_s2          ,
            self.tree.oContYMean_x1_s2          ,
            self.tree.oContYMean_x2_s2          ,
            self.tree.oContYMean_x3_s2          ,
            self.tree.oContYMean_x4_s2          ,
            self.tree.oContYMean_x5_s2          ,
            self.tree.oContLayerMean_x1_s2      ,
            self.tree.oContLayerMean_x2_s2      ,
            self.tree.oContLayerMean_x3_s2      ,
            self.tree.oContLayerMean_x4_s2      ,
            self.tree.oContLayerMean_x5_s2      ,
            self.tree.oContXStd_x1_s2           ,
            self.tree.oContXStd_x2_s2           ,
            self.tree.oContXStd_x3_s2           ,
            self.tree.oContXStd_x4_s2           ,
            self.tree.oContXStd_x5_s2           ,
            self.tree.oContYStd_x1_s2           ,
            self.tree.oContYStd_x2_s2           ,
            self.tree.oContYStd_x3_s2           ,
            self.tree.oContYStd_x4_s2           ,
            self.tree.oContYStd_x5_s2           ,
            self.tree.oContLayerStd_x1_s2       ,
            self.tree.oContLayerStd_x2_s2       ,
            self.tree.oContLayerStd_x3_s2       ,
            self.tree.oContLayerStd_x4_s2       ,
            self.tree.oContLayerStd_x5_s2       ,
            self.tree.oContEnergy_x1_s3         ,
            self.tree.oContEnergy_x2_s3         ,
            self.tree.oContEnergy_x3_s3         ,
            self.tree.oContEnergy_x4_s3         ,
            self.tree.oContEnergy_x5_s3         ,
            self.tree.oContNHits_x1_s3          ,
            self.tree.oContNHits_x2_s3          ,
            self.tree.oContNHits_x3_s3          ,
            self.tree.oContNHits_x4_s3          ,
            self.tree.oContNHits_x5_s3          ,
            self.tree.oContXMean_x1_s3          ,
            self.tree.oContXMean_x2_s3          ,
            self.tree.oContXMean_x3_s3          ,
            self.tree.oContXMean_x4_s3          ,
            self.tree.oContXMean_x5_s3          ,
            self.tree.oContYMean_x1_s3          ,
            self.tree.oContYMean_x2_s3          ,
            self.tree.oContYMean_x3_s3          ,
            self.tree.oContYMean_x4_s3          ,
            self.tree.oContYMean_x5_s3          ,
            self.tree.oContLayerMean_x1_s3      ,
            self.tree.oContLayerMean_x2_s3      ,
            self.tree.oContLayerMean_x3_s3      ,
            self.tree.oContLayerMean_x4_s3      ,
            self.tree.oContLayerMean_x5_s3      ,
            self.tree.oContXStd_x1_s3           ,
            self.tree.oContXStd_x2_s3           ,
            self.tree.oContXStd_x3_s3           ,
            self.tree.oContXStd_x4_s3           ,
            self.tree.oContXStd_x5_s3           ,
            self.tree.oContYStd_x1_s3           ,
            self.tree.oContYStd_x2_s3           ,
            self.tree.oContYStd_x3_s3           ,
            self.tree.oContYStd_x4_s3           ,
            self.tree.oContYStd_x5_s3           ,
            self.tree.oContLayerStd_x1_s3       ,
            self.tree.oContLayerStd_x2_s3       ,
            self.tree.oContLayerStd_x3_s3       ,
            self.tree.oContLayerStd_x4_s3       ,
            self.tree.oContLayerStd_x5_s3       ,
            ]

    # Copy input tree feats to new tree
    for feat_name, feat_value in zip(self.tfMaker.branches_info, feats):
        self.tfMaker.branches[feat_name][0] = feat_value

    # Add prediction to new tree
    evtarray = np.array([feats])
    pred = float(model.predict(xgb.DMatrix(evtarray))[0])
    self.tfMaker.branches['discValue_EcalVeto'][0] = pred

    # Fill new tree with current event values
    self.tfMaker.tree.Fill()


###############
# RUN
###############

if __name__ == '__main__':
    main()
