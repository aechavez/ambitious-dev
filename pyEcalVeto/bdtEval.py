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

def process_event(self):

    # Reset branch values for this tree model
    self.tree_model.reset_values()


###############
# RUN
###############

if __name__ == '__main__':
    main()
