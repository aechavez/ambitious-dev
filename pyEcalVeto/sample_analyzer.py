import numpy as np
import os
import ROOT as r
from modules import mipTracking, physTools
from modules import rootManager as manager


# Load dependencies
cell_map = np.loadtxt('modules/cellModule.txt')
r.gSystem.Load('libFramework.so')

# General ECal branch information
ecal_branch_information = {

    # Event number
    'eventNumber':                {'dtype': int,   'default': 0 },

    # Simulated hit information
    'simHitEnergy':               {'dtype': float, 'default': 0.},
    'nSimHits':                   {'dtype': int,   'default': 0 },

    # Reconstructed hit information
    'recHitAmplitude':            {'dtype': float, 'default': 0.},
    'recHitEnergy':               {'dtype': float, 'default': 0.},
    'nRecHits':                   {'dtype': int,   'default': 0 },

    # Noise information
    'noiseHitEnergy':             {'dtype': float, 'default': 0.},
    'nNoiseHits':                 {'dtype': int,   'default': 0 },

    # Electron kinematics at target scoring plane
    'electronIsAtTargSP':         {'dtype': int,   'default': 0 },
    'electronTargSP_x':           {'dtype': float, 'default': 0.},
    'electronTargSP_y':           {'dtype': float, 'default': 0.},
    'electronTargSP_z':           {'dtype': float, 'default': 0.},
    'electronTargSP_px':          {'dtype': float, 'default': 0.},
    'electronTargSP_py':          {'dtype': float, 'default': 0.},
    'electronTargSP_pz':          {'dtype': float, 'default': 0.},
    'electronTargSP_pmag':        {'dtype': float, 'default': 0.},
    'electronTargSP_ptheta':      {'dtype': float, 'default': 0.},

    # Electron kinematics at ECal scoring plane
    'electronIsAtECalSP':         {'dtype': int,   'default': 0 },
    'electronECalSP_x':           {'dtype': float, 'default': 0.},
    'electronECalSP_y':           {'dtype': float, 'default': 0.},
    'electronECalSP_z':           {'dtype': float, 'default': 0.},
    'electronECalSP_px':          {'dtype': float, 'default': 0.},
    'electronECalSP_py':          {'dtype': float, 'default': 0.},
    'electronECalSP_pz':          {'dtype': float, 'default': 0.},
    'electronECalSP_pmag':        {'dtype': float, 'default': 0.},
    'electronECalSP_ptheta':      {'dtype': float, 'default': 0.},

    # Photon kinematics at target scoring plane
    'photonIsAtTargSP':           {'dtype': int,   'default': 0 },
    'photonTargSP_x':             {'dtype': float, 'default': 0.},
    'photonTargSP_y':             {'dtype': float, 'default': 0.},
    'photonTargSP_z':             {'dtype': float, 'default': 0.},
    'photonTargSP_px':            {'dtype': float, 'default': 0.},
    'photonTargSP_py':            {'dtype': float, 'default': 0.},
    'photonTargSP_pz':            {'dtype': float, 'default': 0.},
    'photonTargSP_pmag':          {'dtype': float, 'default': 0.},
    'photonTargSP_ptheta':        {'dtype': float, 'default': 0.},

    # Photon kinematics at ECal scoring plane
    'photonIsAtECalSP':           {'dtype': int,   'default': 0 },
    'photonECalSP_x':             {'dtype': float, 'default': 0.},
    'photonECalSP_y':             {'dtype': float, 'default': 0.},
    'photonECalSP_z':             {'dtype': float, 'default': 0.},
    'photonECalSP_px':            {'dtype': float, 'default': 0.},
    'photonECalSP_py':            {'dtype': float, 'default': 0.},
    'photonECalSP_pz':            {'dtype': float, 'default': 0.},
    'photonECalSP_pmag':          {'dtype': float, 'default': 0.},
    'photonECalSP_ptheta':        {'dtype': float, 'default': 0.},

    # Fiducial information
    'electronIsFiducial':         {'dtype': int,   'default': 0 },
    'photonIsFiducial':           {'dtype': int,   'default': 0 },

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

for i in range(1, physTools.nregions + 1):

    # Electron RoC variables
    ecal_branch_information['electronContainmentEnergy_x{}'.format(i)] = {'rtype': float, 'default': 0.}

    # Photon RoC variables
    ecal_branch_information['photonContainmentEnergy_x{}'.format(i)]   = {'rtype': float, 'default': 0.}

    # Outside RoC variables
    ecal_branch_information['outsideContainmentEnergy_x{}'.format(i)]  = {'rtype': float, 'default': 0.}
    ecal_branch_information['outsideContainmentNHits_x{}'.format(i)]   = {'rtype': int,   'default': 0 }
    ecal_branch_information['outsideContainmentXStd_x{}'.format(i)]    = {'rtype': float, 'default': 0.}
    ecal_branch_information['outsideContainmentYStd_x{}'.format(i)]    = {'rtype': float, 'default': 0.}

for i in range(1, physTools.nsegments + 1):

    # Longitudinal segment variables
    ecal_branch_information['energy_s{}'.format(i)]    = {'dtype': float, 'default': 0.}
    ecal_branch_information['nHits_s{}'.format(i)]     = {'dtype': int,   'default': 0 }
    ecal_branch_information['xMean_s{}'.format(i)]     = {'dtype': float, 'default': 0.}
    ecal_branch_information['yMean_s{}'.format(i)]     = {'dtype': float, 'default': 0.}
    ecal_branch_information['layerMean_s{}'.format(i)] = {'dtype': float, 'default': 0.}
    ecal_branch_information['xStd_s{}'.format(i)]      = {'dtype': float, 'default': 0.}
    ecal_branch_information['yStd_s{}'.format(i)]      = {'dtype': float, 'default': 0.}
    ecal_branch_information['layerStd_s{}'.format(i)]  = {'dtype': float, 'default': 0.}

    for j in range(1, physTools.nregions + 1):

        # Electron RoC variables
        ecal_branch_information['electronContainmentEnergy_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}
        ecal_branch_information['electronContainmentNHits_x{}_s{}'.format(j, i)]     = {'dtype': int,   'default': 0 }
        ecal_branch_information['electronContainmentXMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        ecal_branch_information['electronContainmentYMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        ecal_branch_information['electronContainmentLayerMean_x{}_s{}'.format(j, i)] = {'dtype': float, 'default': 0.}
        ecal_branch_information['electronContainmentXStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        ecal_branch_information['electronContainmentYStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        ecal_branch_information['electronContainmentLayerStd_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}

        # Photon RoC variables
        ecal_branch_information['photonContainmentEnergy_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        ecal_branch_information['photonContainmentNHits_x{}_s{}'.format(j, i)]       = {'dtype': int,   'default': 0 }
        ecal_branch_information['photonContainmentXMean_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        ecal_branch_information['photonContainmentYMean_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        ecal_branch_information['photonContainmentLayerMean_x{}_s{}'.format(j, i)]   = {'dtype': float, 'default': 0.}
        ecal_branch_information['photonContainmentXStd_x{}_s{}'.format(j, i)]        = {'dtype': float, 'default': 0.}
        ecal_branch_information['photonContainmentYStd_x{}_s{}'.format(j, i)]        = {'dtype': float, 'default': 0.}
        ecal_branch_information['photonContainmentLayerStd_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}

        # Outside RoC variables
        ecal_branch_information['outsideContainmentEnergy_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        ecal_branch_information['outsideContainmentNHits_x{}_s{}'.format(j, i)]      = {'dtype': int,   'default': 0 }
        ecal_branch_information['outsideContainmentXMean_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        ecal_branch_information['outsideContainmentYMean_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        ecal_branch_information['outsideContainmentLayerMean_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}
        ecal_branch_information['outsideContainmentXStd_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        ecal_branch_information['outsideContainmentYStd_x{}_s{}'.format(j, i)]       = {'dtype': float, 'default': 0.}
        ecal_branch_information['outsideContainmentLayerStd_x{}_s{}'.format(j, i)]   = {'dtype': float, 'default': 0.}

# Simulated particle branch informtion
sim_particle_branch_information = {
    'pdgID':      {'dtype': int,   'address': np.zeros(1, dtype = int)  },
    'trackID':    {'dtype': int,   'address': np.zeros(1, dtype = int)  },
    'mass':       {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'charge':     {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'energy':     {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'vert_x':     {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'vert_y':     {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'vert_z':     {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'end_x':      {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'end_y':      {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'end_z':      {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'px':         {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'py':         {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'pz':         {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'nParents':   {'dtype': int,   'address': np.zeros(1, dtype = int)  },
    'nDaughters': {'dtype': int,   'address': np.zeros(1, dtype = int)  }
}
for branch_name in ecal_branch_information:
    sim_particle_branch_information[branch_name] = {'dtype': ecal_branch_information[branch_name]['dtype'],\
                                                    'address': np.zeros(1, dtype = ecal_branch_information[branch_name]['dtype'])}

# Simulated hit branch informtion
sim_hit_branch_information = {
    'energy': {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'x':      {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'y':      {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'z':      {'dtype': float, 'address': np.zeros(1, dtype = float)}
}
for branch_name in ecal_branch_information:
    sim_hit_branch_information[branch_name] = {'dtype': ecal_branch_information[branch_name]['dtype'],\
                                               'address': np.zeros(1, dtype = ecal_branch_information[branch_name]['dtype'])}

# Reconstructed hit branch information
rec_hit_branch_information = {
    'amplitude':         {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'energy':            {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'matchingSimEnergy': {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'x':                 {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'y':                 {'dtype': float, 'address': np.zeros(1, dtype = float)},
    'z':                 {'dtype': float, 'address': np.zeros(1, dtype = float)}
}
for branch_name in ecal_branch_information:
    rec_hit_branch_information[branch_name] = {'dtype': ecal_branch_information[branch_name]['dtype'],\
                                               'address': np.zeros(1, dtype = ecal_branch_information[branch_name]['dtype'])}


###########################
# Main subroutine
###########################

def main():

    # Parse arguments passed by the user
    parsing_dict = manager.parse()
    batch_mode = parsing_dict['batch_mode']
    separate_categories = parsing_dict['separate_categories']
    inputs = parsing_dict['inputs']
    group_labels = parsing_dict['group_labels']
    outputs = parsing_dict['outputs']
    start_event = parsing_dict['start_event']
    max_events = parsing_dict['max_events']

    # Build a tree process for each file group
    processes = []
    for label, group in zip(group_labels, inputs):
        processes.append(manager.TreeProcess(process_event, file_group = group, name_tag = label, batch_mode = batch_mode))

    # Loop to prepare each process and run
    for process in processes:

        # Move to the temporary directory for this process
        os.chdir(process.temporary_directory)

        # Add branches needed for analysis
        process.event_header = process.addBranch('EventHeader', 'EventHeader')
        process.target_sp_hits = process.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        process.ecal_sp_hits = process.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        process.sim_particles = process.addBranch('SimParticle', 'SimParticles_v12')
        process.ecal_sim_hits = process.addBranch('SimCalorimeterHit', 'EcalSimHits_v12')
        process.ecal_rec_hits = process.addBranch('EcalHit', 'EcalRecHits_v12')
        process.ecal_veto = process.addBranch('EcalVetoResult', 'EcalVeto_v12')

        # Dictionary of categories for tree models
        process.separate_categories = separate_categories
        if process.separate_categories:
            process.tree_models = {
                'fiducial_electron_photon': None,
                'fiducial_electron': None,
                'fiducial_photon': None,
                'non_fiducial': None
            }
        else: process.tree_models = {'unsorted': None}

        # Build a tree model for each category
        for tree_model in process.tree_models:
            process.tree_models[tree_model] = manager.TreeMaker('{}_{}.root'.format(group_labels[processes.index(process)], tree_model),\
                                                                'EcalVeto', branch_information = branch_information,\
                                                                out_directory = outputs[processes.index(process)])

        # Tree for simulated particle information
        process.sim_particles_tree = r.TTree('SimParticles', 'Simulated particle information')

        for branch_name in sim_particle_branch_information:

            if str(sim_particle_branch_information[branch_name]['dtype']) == "<class 'float'>"\
              or str(sim_particle_branch_information[branch_name]['dtype']) == "<type 'float'>":
                process.sim_particles_tree.Branch(branch_name, sim_particle_branch_information[branch_name]['address'], '{}/D'.format(branch_name))

            elif str(sim_particle_branch_information[branch_name]['dtype']) == "<class 'int'>"\
              or str(sim_particle_branch_information[branch_name]['dtype']) == "<type 'int'>":
                process.sim_particles_tree.Branch(branch_name, sim_particle_branch_information[branch_name]['address'], '{}/I'.format(branch_name))

        # Tree for simulated hit information
        process.sim_hits_tree = r.TTree('SimHits', 'Simulated hit information')

        for branch_name in sim_hit_branch_information:

            if str(sim_hit_branch_information[branch_name]['dtype']) == "<class 'float'>"\
              or str(sim_hit_branch_information[branch_name]['dtype']) == "<type 'float'>":
                process.sim_hits_tree.Branch(branch_name, sim_hit_branch_information[branch_name]['address'], '{}/D'.format(branch_name))

            elif str(sim_hit_branch_information[branch_name]['dtype']) == "<class 'int'>"\
              or str(sim_hit_branch_information[branch_name]['dtype']) == "<type 'int'>":
                process.sim_hits_tree.Branch(branch_name, sim_hit_branch_information[branch_name]['address'], '{}/I'.format(branch_name))

        # Tree for reconstructed hit information
        process.rec_hits_tree = r.TTree('RecHits', 'Reconstructed hit information')

        for branch_name in rec_hit_branch_information:

            if str(rec_hit_branch_information[branch_name]['dtype']) == "<class 'float'>"\
              or str(rec_hit_branch_information[branch_name]['dtype']) == "<type 'float'>":
                process.rec_hits_tree.Branch(branch_name, rec_hit_branch_information[branch_name]['address'], '{}/D'.format(branch_name))

            elif str(rec_hit_branch_information[branch_name]['dtype']) == "<class 'int'>"\
              or str(rec_hit_branch_information[branch_name]['dtype']) == "<type 'int'>":
                process.rec_hits_tree.Branch(branch_name, rec_hit_branch_information[branch_name]['address'], '{}/I'.format(branch_name))

        # Set closing functions
        process.closing_functions = []

        for tree_model in process.tree_models:
            process.closing_functions.append(process.sim_particles_tree.Write)
            process.closing_functions.append(process.sim_hits_tree.Write)
            process.closing_functions.append(process.rec_hits_tree.Write)

        for tree_model in process.tree_models:
            process.closing_functions.append(process.tree_models[tree_model].write)

        # Run this process
        process.run(start_event = start_event, max_events = max_events, print_frequency = 100)

    # Remove the scratch directory (Being careful not to break other jobs)
    if not batch_mode: manager.remove_scratch()

    print('\n[ INFO ] - All processes finished!')


##########################################
# Subroutine to process an event
##########################################

def process_event(self):

    # Reset branch values for each tree model
    for tree_model in self.tree_models:
        self.tree_models[tree_model].reset_values()

    # Dictionary of new values
    new_values = {branch_name: branch_information[branch_name]['default'] for branch_name in branch_information}


    ###########################################
    # Electron and photon information
    ###########################################

    # Get the electron at the target scoring plane
    ele_target_sp_hit = physTools.get_ele_targ_sp_hit(self.target_sp_hits)

    if not (ele_target_sp_hit is None):
        ele_target_sp_pos = physTools.get_position(ele_target_sp_hit)
        ele_target_sp_mom = physTools.get_momentum(ele_target_sp_hit)
    else:
        print('[ WARNING ] - No electron at target scoring plane!')
        ele_target_sp_pos = ele_target_sp_mom = np.zeros(3)

    # Get the electron at the ECal scoring plane
    ele_ecal_sp_hit = physTools.get_ele_ecal_sp_hit(self.ecal_sp_hits)

    if not (ele_ecal_sp_hit is None):
        ele_ecal_sp_pos = physTools.get_position(ele_ecal_sp_hit)
        ele_ecal_sp_mom = physTools.get_momentum(ele_ecal_sp_hit)
    else:
        print('[ WARNING ] - No electron at ECal scoring plane!')
        ele_ecal_sp_pos = ele_ecal_sp_mom = np.zeros(3)

    # Infer the photon's information at the target scoring plane
    if not (ele_target_sp_hit is None):
        pho_target_sp_pos, pho_target_sp_mom = physTools.infer_pho_targ_sp_hit(ele_target_sp_hit)
    else:
        pho_target_sp_pos = pho_target_sp_mom = np.zeros(3)

    # Use linear projections to infer the trajectories
    ele_traj = pho_traj = None

    if not (ele_ecal_sp_hit is None):
        ele_traj = physTools.get_intercepts(ele_ecal_sp_pos, ele_ecal_sp_mom, physTools.ecal_layerZs)

    if not (ele_target_sp_hit is None):
        pho_traj = physTools.get_intercepts(pho_target_sp_pos, pho_target_sp_mom, physTools.ecal_layerZs)

    # Determine which fiducial category the event belongs to
    if self.separate_categories:
        fid_ele = fid_pho = False

        if not (ele_traj is None):
            for cell in cell_map:
                if physTools.distance(np.array(cell[1:]), ele_traj[0]) <= physTools.cell_radius:
                    fid_ele = True
                    break

        if not (pho_traj is None):
            for cell in cell_map:
                if physTools.distance(np.array(cell[1:]), pho_traj[0]) <= physTools.cell_radius:
                    fid_pho = True
                    break


    ###########################################
    # Assign pre-calculated variables
    ###########################################

    new_values['nReadoutHits'] = self.ecal_veto.getNReadoutHits()
    new_values['summedDet'] = self.ecal_veto.getSummedDet()
    new_values['summedTightIso'] = self.ecal_veto.getSummedTightIso()
    new_values['maxCellDep'] = self.ecal_veto.getMaxCellDep()
    new_values['showerRMS'] = self.ecal_veto.getShowerRMS()
    new_values['xStd'] = self.ecal_veto.getXStd()
    new_values['yStd'] = self.ecal_veto.getYStd()
    new_values['avgLayerHit'] = self.ecal_veto.getAvgLayerHit()
    new_values['stdLayerHit'] = self.ecal_veto.getStdLayerHit()
    new_values['deepestLayerHit'] = self.ecal_veto.getDeepestLayerHit() 
    new_values['ecalBackEnergy'] = self.ecal_veto.getEcalBackEnergy()

    for i in range(0, physTools.nregions):
        new_values['electronContainmentEnergy_x{}'.format(i + 1)] = self.ecalVeto.getElectronContainmentEnergy()[i]
        new_values['photonContainmentEnergy_x{}'.format(i + 1)] = self.ecalVeto.getPhotonContainmentEnergy()[i]
        new_values['outsideContainmentEnergy_x{}'.format(i + 1)] = self.ecalVeto.getOutsideContainmentEnergy()[i]
        new_values['outsideContainmentNHits_x{}'.format(i + 1)] = self.ecalVeto.getOutsideContainmentNHits()[i]
        new_values['outsideContainmentXStd_x{}'.format(i + 1)] = self.ecalVeto.getOutsideContainmentXStd()[i]
        new_values['outsideContainmentYStd_x{}'.format(i + 1)] = self.ecalVeto.getOutsideContainmentYStd()[i]


    ###############################
    # MIP tracking set up
    ###############################

    # Get endpoints of each trajectory and calculate trajectory variables
    if not ((ele_traj is None) or (pho_traj is None)):

        # Arrays marking endpoints of each trajectory
        ele_traj_ends = np.array([[ele_traj[0][0], ele_traj[0][1], physTools.ecal_layerZs[0]],\
                                  [ele_traj[-1][0], ele_traj[-1][1], physTools.ecal_layerZs[-1]]])
        pho_traj_ends = np.array([[pho_traj[0][0], pho_traj[0][1], physTools.ecal_layerZs[0]],\
                                  [pho_traj[-1][0], pho_traj[-1][1], physTools.ecal_layerZs[-1]]])

        ele_traj_vec = physTools.normalize(ele_traj_ends[1] - ele_traj_ends[0])
        pho_traj_vec = physTools.normalize(pho_traj_ends[1] - pho_traj_ends[0])

        new_values['trajSep'] = physTools.distance(ele_traj_ends[0], pho_traj_ends[0])
        new_values['trajDot'] = np.dot(ele_traj_vec, pho_traj_vec)
    else:

        # One of the trajectories is missing, so use all of the hits in the ECal for MIP tracking
        # Take endpoints far outside the ECal so they don't restrict tracking
        ele_traj_ends = np.array([[999., 999., 0.], [999., 999., 999.]])
        pho_traj_ends = np.array([[1000., 1000., 0.], [1000., 1000., 1000.]])

        # Assign dummy values in this case
        new_values['trajSep'] = 11.
        new_values['trajDot'] = 4.

    # For straight tracks algorithm
    tracking_hit_list = []

    # For territory variables
    pho_to_ele = physTools.normalize(ele_traj_ends[0] - pho_traj_ends[0])
    origin = 0.5*physTools.cell_width*pho_to_ele + pho_traj_ends[0]


    ######################
    # RoC set up
    ######################

    # Recoil momentum magnitude and angle
    recoil_mom_mag = recoil_mom_theta = -1

    if not (ele_ecal_sp_hit is None):
        recoil_mom_mag = np.linalg.norm(ele_ecal_sp_mom)
        recoil_mom_theta = physTools.angle(ele_ecal_sp_mom, np.array([0, 0, 1]), units = 'degrees')

    # Set electron RoC binnings
    ele_radii = physTools.radius68_thetalt10_plt500
    if recoil_mom_theta < 10 and recoil_mom_mag >= 500: ele_radii = physTools.radius68_thetalt10_pgt500
    elif recoil_mom_theta >= 10 and recoil_mom_theta < 20: ele_radii = physTools.radius68_theta10to20
    elif recoil_mom_theta >= 20: ele_radii = physTools.radius68_thetagt20

    # Use default binning for photon RoC
    pho_radii = physTools.radius68_thetalt10_plt500


    ###########################
    # Major ECal loop
    ###########################

    for hit in self.ecal_rec_hits:
        
        if hit.getEnergy() > 0:

            layer = physTools.get_ecal_layer(hit)
            xy_pos = physTools.get_position(hit)[0:2]

            # Distances to inferred trajectories
            dist_ele_traj = dist_pho_traj = -1

            if not (ele_traj is None):
                xy_ele_traj = np.array([ele_traj[layer][0], ele_traj[layer][1]])
                dist_ele_traj = physTools.distance(xy_pos, xy_ele_traj)

            if not (pho_traj is None):
                xy_pho_traj = np.array([pho_traj[layer][0], pho_traj[layer][1]])
                dist_pho_traj = physTools.distance(xy_pos, xy_pho_traj)

            # Determine which full territory the hit is in and add to sums
            hit_prime = physTools.get_position(hit) - origin

            if np.dot(hit_prime, pho_to_ele) > 0: new_values['nFullElectronTerritoryHits'] += 1
            else:  new_values['nFullPhotonTerritoryHits'] += 1

            # Determine which longitudinal segment the hit is in and add to sums
            for i in range(1, physTools.nsegments + 1):

                if physTools.segment_ends[i - 1][0] <= layer and layer <= physTools.segment_ends[i - 1][1]:
                    new_values['energy_s{}'.format(i)] += hit.getEnergy()
                    new_values['nHits_s{}'.format(i)] += 1
                    new_values['xMean_s{}'.format(i)] += xy_pos[0]*hit.getEnergy()
                    new_values['yMean_s{}'.format(i)] += xy_pos[1]*hit.getEnergy()
                    new_values['layerMean_s{}'.format(i)] += layer*hit.getEnergy()

                    # Determine which containment region the hit is in and add to sums
                    for j in range(1, physTools.nregions + 1):

                        if (j - 1)*ele_radii[layer] <= dist_ele_traj and dist_ele_traj < j*ele_radii[layer]:
                            new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)] += hit.getEnergy()
                            new_values['electronContainmentNHits_x{}_s{}'.format(j, i)] += 1
                            new_values['electronContainmentXMean_x{}_s{}'.format(j, i)] += xy_pos[0]*hit.getEnergy()
                            new_values['electronContainmentYMean_x{}_s{}'.format(j, i)] += xy_pos[1]*hit.getEnergy()
                            new_values['electronContainmentLayerMean_x{}_s{}'.format(j, i)] += layer*hit.getEnergy()

                        if (j - 1)*pho_radii[layer] <= dist_pho_traj and dist_pho_traj < j*pho_radii[layer]:
                            new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)] += hit.getEnergy()
                            new_values['photonContainmentNHits_x{}_s{}'.format(j, i)] += 1
                            new_values['photonContainmentXMean_x{}_s{}'.format(j, i)] += xy_pos[0]*hit.getEnergy()
                            new_values['photonContainmentYMean_x{}_s{}'.format(j, i)] += xy_pos[1]*hit.getEnergy()
                            new_values['photonContainmentLayerMean_x{}_s{}'.format(j, i)] += layer*hit.getEnergy()

                        if dist_ele_traj > j*ele_radii[layer] and dist_pho_traj > j*pho_radii[layer]:
                            new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)] += hit.getEnergy()
                            new_values['outsideContainmentNHits_x{}_s{}'.format(j, i)] += 1
                            new_values['outsideContainmentXMean_x{}_s{}'.format(j, i)] += xy_pos[0]*hit.getEnergy()
                            new_values['outsideContainmentYMean_x{}_s{}'.format(j, i)] += xy_pos[1]*hit.getEnergy()
                            new_values['outsideContainmentLayerMean_x{}_s{}'.format(j, i)] += layer*hit.getEnergy()

            # Add to MIP tracking hit list if outside electron RoC or electron trajectory is missing
            if dist_ele_traj >= ele_radii[layer] or dist_ele_traj == -1:
                tracking_hit_list.append(hit) 

    # Quotient out the total energy from the means if possible
    for i in range(1, physTools.nsegments + 1):

        if new_values['energy_s{}'.format(i)] > 0:
            new_values['xMean_s{}'.format(i)] /= new_values['energy_s{}'.format(i)]
            new_values['yMean_s{}'.format(i)] /= new_values['energy_s{}'.format(i)]
            new_values['layerMean_s{}'.format(i)] /= new_values['energy_s{}'.format(i)]

        for j in range(1, physTools.nregions + 1):

            if new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)] > 0:
                new_values['electronContainmentXMean_x{}_s{}'.format(j, i)] /= new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)]
                new_values['electronContainmentYMean_x{}_s{}'.format(j, i)] /= new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)]
                new_values['electronContainmentLayerMean_x{}_s{}'.format(j, i)] /= new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)]

            if new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)] > 0:
                new_values['photonContainmentXMean_x{}_s{}'.format(j, i)] /= new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)]
                new_values['photonContainmentYMean_x{}_s{}'.format(j, i)] /= new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)]
                new_values['photonContainmentLayerMean_x{}_s{}'.format(j, i)] /= new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)]

            if new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)] > 0:
                new_values['outsideContainmentXMean_x{}_s{}'.format(j, i)] /= new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)]
                new_values['outsideContainmentYMean_x{}_s{}'.format(j, i)] /= new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)]
                new_values['outsideContainmentLayerMean_x{}_s{}'.format(j, i)] /= new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)]

    # Loop over hits again to calculate standard deviations
    for hit in self.ecal_rec_hits:

        layer = physTools.get_ecal_layer(hit)
        xy_pos = physTools.get_position(hit)[0:2]

        # Distances to inferred trajectories
        dist_ele_traj = dist_pho_traj = -1

        if not (ele_traj is None):
            xy_ele_traj = np.array([ele_traj[layer][0], ele_traj[layer][1]])
            dist_ele_traj = physTools.distance(xy_pos, xy_ele_traj)

        if not (pho_traj is None):
            xy_pho_traj = np.array([pho_traj[layer][0], pho_traj[layer][1]])
            dist_pho_traj = physTools.distance(xy_pos, xy_pho_traj)

        # Determine which longitudinal segment the hit is in and add to sums
        for i in range(1, physTools.nsegments + 1):

            if physTools.segment_ends[i - 1][0] <= layer and layer <= physTools.segment_ends[i - 1][1]:
                new_values['xStd_s{}'.format(i)] += ((xy_pos[0] - new_values['xMean_s{}'.format(i)])**2)*hit.getEnergy()
                new_values['yStd_s{}'.format(i)] += ((xy_pos[1] - new_values['yMean_s{}'.format(i)])**2)*hit.getEnergy()
                new_values['layerStd_s{}'.format(i)] += ((layer - new_values['layerMean_s{}'.format(i)])**2)*hit.getEnergy()

                # Determine which containment region the hit is in and add to sums
                for j in range(1, physTools.nregions + 1):

                    if (j - 1)*ele_radii[layer] <= dist_ele_traj and dist_ele_traj < j*ele_radii[layer]:
                        new_values['electronContainmentXStd_x{}_s{}'.format(j, i)] +=\
                        ((xy_pos[0] - new_values['electronContainmentXMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()
                        new_values['electronContainmentYStd_x{}_s{}'.format(j, i)] +=\
                        ((xy_pos[1] - new_values['electronContainmentYMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()
                        new_values['electronContainmentLayerStd_x{}_s{}'.format(j, i)] +=\
                        ((layer - new_values['electronContainmentLayerMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()

                    if (j - 1)*pho_radii[layer] <= dist_pho_traj and dist_pho_traj < j*pho_radii[layer]:
                        new_values['photonContainmentXStd_x{}_s{}'.format(j, i)] +=\
                        ((xy_pos[0] - new_values['photonContainmentXMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()
                        new_values['photonContainmentYStd_x{}_s{}'.format(j, i)] +=\
                        ((xy_pos[1] - new_values['photonContainmentYMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()
                        new_values['photonContainmentLayerStd_x{}_s{}'.format(j, i)] +=\
                        ((layer - new_values['photonContainmentLayerMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()

                    if dist_ele_traj > j*ele_radii[layer] and dist_pho_traj > j*pho_radii[layer]:
                        new_values['outsideContainmentXStd_x{}_s{}'.format(j, i)] +=\
                        ((xy_pos[0] - new_values['outsideContainmentXMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()
                        new_values['outsideContainmentYStd_x{}_s{}'.format(j, i)] +=\
                        ((xy_pos[1] - new_values['outsideContainmentYMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()
                        new_values['outsideContainmentLayerStd_x{}_s{}'.format(j, i)] +=\
                        ((layer - new_values['outsideContainmentLayerMean_x{}_s{}'.format(j, i)])**2)*hit.getEnergy()

    # Quotient out the total energies from the standard deviations if possible and take root
    for i in range(1, physTools.nsegments + 1):

        if new_values['energy_s{}'.format(i)] > 0:
            new_values['xStd_s{}'.format(i)] = np.sqrt(new_values['xStd_s{}'.format(i)]/new_values['energy_s{}'.format(i)])
            new_values['yStd_s{}'.format(i)] = np.sqrt(new_values['yStd_s{}'.format(i)]/new_values['energy_s{}'.format(i)])
            new_values['layerStd_s{}'.format(i)] = np.sqrt(new_values['layerStd_s{}'.format(i)]/new_values['energy_s{}'.format(i)])

        for j in range(1, physTools.nregions + 1):

            if new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)] > 0:
                new_values['electronContainmentXStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['electronContainmentXStd_x{}_s{}'.format(j, i)]/new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)])
                new_values['electronContainmentYStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['electronContainmentYStd_x{}_s{}'.format(j, i)]/new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)])
                new_values['electronContainmentLayerStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['electronContainmentLayerStd_x{}_s{}'.format(j, i)]/new_values['electronContainmentEnergy_x{}_s{}'.format(j, i)])

            if new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)] > 0:
                new_values['photonContainmentXStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['photonContainmentXStd_x{}_s{}'.format(j, i)]/new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)])
                new_values['photonContainmentYStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['photonContainmentYStd_x{}_s{}'.format(j, i)]/new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)])
                new_values['photonContainmentLayerStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['photonContainmentLayerStd_x{}_s{}'.format(j, i)]/new_values['photonContainmentEnergy_x{}_s{}'.format(j, i)])

            if new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)] > 0:
                new_values['outsideContainmentXStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['outsideContainmentXStd_x{}_s{}'.format(j, i)]/new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)])
                new_values['outsideContainmentYStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['outsideContainmentYStd_x{}_s{}'.format(j, i)]/new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)])
                new_values['outsideContainmentLayerStd_x{}_s{}'.format(j, i)] =\
                np.sqrt(new_values['outsideContainmentLayerStd_x{}_s{}'.format(j, i)]/new_values['outsideContainmentEnergy_x{}_s{}'.format(j, i)])


    ########################
    # MIP tracking
    ########################

    # Calculate near photon variables
    if not (pho_traj is None):
        new_values['firstNearPhotonLayer'], new_values['nNearPhotonHits'] = mipTracking.nearPhotonInfo(tracking_hit_list, pho_traj)
    else: new_values['nNearPhotonHits'] = new_values['nReadoutHits']

    # Determine which territory the hit is in and add to sums
    if not (ele_traj is None):
        for hit in tracking_hit_list:
            hit_prime = physTools.get_position(hit) - origin
            if np.dot(hit_prime, pho_to_ele) > 0: new_values['nElectronTerritoryHits'] += 1
            else: new_values['nPhotonTerritoryHits'] += 1
    else:
        new_values['nPhotonTerritoryHits'] = new_values['nReadoutHits']
        new_values['territoryRatio'] = 10
        new_values['nFullTerritoryRatio'] = 10

    if new_values['nElectronTerritoryHits'] != 0:
        new_values['territoryRatio'] = new_values['nPhotonTerritoryHits']/new_values['nElectronTerritoryHits']
    if new_values['nFullElectronTerritoryHits'] != 0:
        new_values['fullTerritoryRatio'] = new_values['nFullPhotonTerritoryHits']/new_values['nFullElectronTerritoryHits']

    # Find straight tracks
    new_values['nStraightTracks'], tracking_hit_list = mipTracking.findStraightTracks(tracking_hit_list, ele_traj_ends,\
                                                                                      pho_traj_ends, mst = 4, returnHitList = True)


    ########################################################
    # General ECal information for sample analysis
    ########################################################

    # Event number
    new_values['eventNumber'] = self.eventHeader.getEventNumber()

    # Simulated hit information
    for hit in self.ecal_sim_hits:
        new_values['simHitEnergy'] += hit.getEdep()
        new_values['nSimHits'] += 1

    # Reconstructed hit information
    for hit in self.ecal_rec_hits:
        new_values['recHitAmplitude'] += hit.getAmplitude()
        new_values['recHitEnergy'] += hit.getEnergy()
        new_values['nRecHits'] += 1

    # Sorted copies of reconstructed/simulated hits to use some shortcuts
    sorted_ecal_rec_hits = [hit for hit in self.ecal_rec_hits]
    sorted_ecal_sim_hits = [hit for hit in self.ecal_sim_hits]
    sorted_ecal_rec_hits.sort(key = lambda hit : hit.getID())
    sorted_ecal_sim_hits.sort(key = lambda hit : hit.getID())

    # Noise hit information
    for rec_hit in sorted_ecal_rec_hits:

        # Count the hit as a noise hit if the noise flag is set
        if rec_hit.isNoise():
            new_values['noiseHitEnergy'] += rec_hit.getEnergy()
            new_values['nNoiseHits'] += 1

        # Otherwise check for a simulated hit whose ID matches
        else:

            nmatch = 0
            for sim_hit in sorted_ecal_sim_hits:
                if sim_hit.getID() == rec_hit.getID(): nmatch += 1
                elif sim_hit.getID() > rec_hit.getID(): break

            # Count the hit as a noise hit if no matching hit exists
            if nmatch == 0:
                new_values['noiseHitEnergy'] += rec_hit.getEnergy()
                new_values['nNoiseHits'] += 1

    # Electron kinematics at the target scoring plane
    if not (ele_target_sp_hit is None):

        new_values['electronIsAtTargSP'] = 1
        new_values['electronTargSP_x'] = ele_target_sp_pos[0]
        new_values['electronTargSP_y'] = ele_target_sp_pos[1]
        new_values['electronTargSP_z'] = ele_target_sp_pos[2]
        new_values['electronTargSP_px'] = ele_target_sp_mom[0]
        new_values['electronTargSP_py'] = ele_target_sp_mom[1]
        new_values['electronTargSP_pz'] = ele_target_sp_mom[2]
        new_values['electronTargSP_pmag'] = np.linalg.norm(ele_target_sp_mom)
        new_values['electronTargSP_ptheta'] = physTools.angle(ele_target_sp_mom, np.array([0, 0, 1]), unit = 'degrees')

    # Electron kinematics at the ECal scoring plane
    if not (ele_ecal_sp_hit is None):

        new_values['electronIsAtECalSP'] = 1
        new_values['electronECalSP_x'] = ele_ecal_sp_pos[0]
        new_values['electronECalSP_y'] = ele_ecal_sp_pos[1]
        new_values['electronECalSP_z'] = ele_ecal_sp_pos[2]
        new_values['electronECalSP_px'] = ele_ecal_sp_mom[0]
        new_values['electronECalSP_py'] = ele_ecal_sp_mom[1]
        new_values['electronECalSP_pz'] = ele_ecal_sp_mom[2]
        new_values['electronECalSP_pmag'] = np.linalg.norm(ele_ecal_sp_mom)
        new_values['electronECalSP_ptheta'] = physTools.angle(ele_ecal_sp_mom, np.array([0, 0, 1]), unit = 'degrees')

    # Get the photon at the target scoring plane
    pho_target_sp_hit_true = physTools.get_pho_targ_sp_hit(self.target_sp_hits)

    if not (pho_target_sp_hit_true is None):
        pho_target_sp_pos_true = physTools.get_position(pho_target_sp_hit_true)
        pho_target_sp_mom_true = physTools.get_momentum(pho_target_sp_hit_true)
    else:
        pho_target_sp_pos_true = pho_target_sp_mom_true = np.zeros(3)

    # Get the photon at the ECal scoring plane
    pho_ecal_sp_hit_true = physTools.get_pho_ecal_sp_hit(self.ecal_sp_hits)

    if not (pho_ecal_sp_hit_true is None):
        pho_ecal_sp_pos_true = physTools.get_position(pho_ecal_sp_hit_true)
        pho_ecal_sp_mom_true = physTools.get_momentum(pho_ecal_sp_hit_true)
    else:
        pho_ecal_sp_pos_true = pho_ecal_sp_mom_true = np.zeros(3)

    # Photon kinematics at the target scoring plane
    if not (pho_target_sp_hit_true is None):

        new_values['photonIsAtTargSP'] = 1
        new_values['photonTargSP_x'] = pho_target_sp_pos_true[0]
        new_values['photonTargSP_y'] = pho_target_sp_pos_true[1]
        new_values['photonTargSP_z'] = pho_target_sp_pos_true[2]
        new_values['photonTargSP_px'] = pho_target_sp_mom_true[0]
        new_values['photonTargSP_py'] = pho_target_sp_mom_true[1]
        new_values['photonTargSP_pz'] = pho_target_sp_mom_true[2]
        new_values['photonTargSP_pmag'] = np.linalg.norm(pho_target_sp_mom_true)
        new_values['photonTargSP_ptheta'] = physTools.angle(pho_target_sp_mom_true, np.array([0, 0, 1]), unit = 'degrees')

    # Photon kinematics at the ECal scoring plane
    if not (pho_ecal_sp_hit_true is None):

        new_values['photonIsAtECalSP'] = 1
        new_values['photonECalSP_x'] = pho_ecal_sp_pos_true[0]
        new_values['photonECalSP_y'] = pho_ecal_sp_pos_true[1]
        new_values['photonECalSP_z'] = pho_ecal_sp_pos_true[2]
        new_values['photonECalSP_px'] = pho_ecal_sp_mom_true[0]
        new_values['photonECalSP_py'] = pho_ecal_sp_mom_true[1]
        new_values['photonECalSP_pz'] = pho_ecal_sp_mom_true[2]
        new_values['photonECalSP_pmag'] = np.linalg.norm(pho_ecal_sp_mom_true)
        new_values['photonECalSP_ptheta'] = physTools.angle(pho_ecal_sp_mom_true, np.array([0, 0, 1]), unit = 'degrees')

    # Electron fiducial information
    fid_ele_true = 0

    if not (ele_ecal_sp_hit is None):

        slope_xz = 999
        if ele_ecal_sp_mom[0] != 0:
            slope_xz = ele_ecal_sp_mom[2]/ele_ecal_sp_mom[0]

        fx = (physTools.ecal_layerZs[0] - ele_ecal_sp_pos[2])/slope_xz + ele_ecal_sp_pos[0]

        slope_yz = 999
        if ele_ecal_sp_mom[1] != 0:
            slope_yz = ele_ecal_sp_mom[2]/ele_ecal_sp_mom[1]

        fy = (physTools.ecal_layerZs[0] - ele_ecal_sp_pos[2])/slope_yz + ele_ecal_sp_pos[1]

        for cell in cell_map:

            cell_dist = physTools.distance(np.array([fx, fy]), np.array([cell[1], cell[2]]))

            if cell_dist <= physTools.cell_radius:
                fid_ele_true = 1
                break

    new_values['electronIsFiducial'] = fid_ele_true

    # Photon fiducial information
    fid_pho_true = 0

    if not (pho_ecal_sp_hit_true is None):

        slope_xz = 999
        if pho_ecal_sp_mom_true[0] != 0:
            slope_xz = pho_ecal_sp_mom_true[2]/pho_ecal_sp_mom_true[0]

        fx = (physTools.ecal_layerZs[0] - pho_ecal_sp_pos_true[2])/slope_xz + pho_ecal_sp_pos_true[0]

        slope_yz = 999
        if pho_ecal_sp_mom_true[1] != 0:
            slope_yz = pho_ecal_sp_mom_true[2]/pho_ecal_sp_mom_true[1]

        fy = (physTools.ecal_layerZs[0] - pho_ecal_sp_pos_true[2])/slope_yz + pho_ecal_sp_pos_true[1]

        for cell in cell_map:

            cell_dist = physTools.distance(np.array([fx, fy]), np.array([cell[1], cell[2]]))

            if cell_dist <= physTools.cell_radius:
                fid_pho_true = 1
                break

    new_values['photonIsFiducial'] = fid_pho_true


    ##############################################################
    # Simulated particle information for sample analysis
    ##############################################################

    for ID, particle in self.sim_particles:

        sim_particle_branch_information['pdgID']['address'][0] = particle.getPdgID()
        sim_particle_branch_information['trackID']['address'][0] = ID
        sim_particle_branch_information['mass']['address'][0] = particle.getMass()
        sim_particle_branch_information['charge']['address'][0] = particle.getCharge()
        sim_particle_branch_information['energy']['address'][0] = particle.getEnergy()
        sim_particle_branch_information['vert_x']['address'][0] = particle.getVertex()[0]
        sim_particle_branch_information['vert_y']['address'][0] = particle.getVertex()[1]
        sim_particle_branch_information['vert_z']['address'][0] = particle.getVertex()[2]
        sim_particle_branch_information['end_x']['address'][0] = particle.getEndPoint()[0]
        sim_particle_branch_information['end_y']['address'][0] = particle.getEndPoint()[1]
        sim_particle_branch_information['end_z']['address'][0] = particle.getEndPoint()[2]
        sim_particle_branch_information['px']['address'][0] = particle.getMomentum()[0]
        sim_particle_branch_information['py']['address'][0] = particle.getMomentum()[1]
        sim_particle_branch_information['pz']['address'][0] = particle.getMomentum()[2]
        sim_particle_branch_information['nParents']['address'][0] = len(particle.getParents())
        sim_particle_branch_information['nDaughters']['address'][0] = len(particle.getDaughters())

        for branch_name in ecal_branch_information:
            sim_particle_branch_information[branch_name]['address'][0] = new_values[branch_name]

        self.sim_particles_tree.Fill()


    #########################################################
    # Simulated hit information for sample analysis
    #########################################################

    for hit in self.ecal_sim_hits:

        sim_hit_branch_information['energy']['address'][0] = hit.getEdep()
        sim_hit_branch_information['x']['address'][0] = hit.getPosition()[0]
        sim_hit_branch_information['y']['address'][0] = hit.getPosition()[1]
        sim_hit_branch_information['z']['address'][0] = hit.getPosition()[2]

        for branch_name in ecal_branch_information:
            sim_hit_branch_information[branch_name]['address'][0] = new_values[branch_name]

        self.sim_hits_tree.Fill()


    #############################################################
    # Reconstructed hit information for sample analysis
    #############################################################

    for hit in self.ecal_rec_hits:

        rec_hit_branch_information['amplitude']['address'][0] = hit.getAmplitude()
        rec_hit_branch_information['energy']['address'][0] = hit.getEnergy()
        rec_hit_branch_information['x']['address'][0] = hit.getXPos()
        rec_hit_branch_information['y']['address'][0] = hit.getYPos()
        rec_hit_branch_information['z']['address'][0] = hit.getZPos()

        matching_edep = 0
        for sim_hit in self.ecal_sim_hits:

            if sim_hit.getID() == hit.getID():
                matching_edep += sim_hit.getEdep()

        rec_hit_branch_information['matchingSimEnergy']['address'][0] = matching_edep

        for branch_name in ecal_branch_information:
            rec_hit_branch_information[branch_name]['address'][0] = new_values[branch_name]

        self.rec_hits_tree.Fill()


    ######################
    # Fill trees
    ######################

    # Fill the branches of each tree model with new values
    if not self.separate_categories:
        self.tree_models['unsorted'].fill(new_values)
    else:
        if fid_ele and fid_pho: self.tree_models['fiducial_electron_photon'].fill(new_values)
        elif fid_ele and (not fid_pho): self.tree_models['fiducial_electron'].fill(new_values)
        elif (not fid_ele) and fid_pho: self.tree_models['fiducial_photon'].fill(new_values)
        else: self.tree_models['non_fiducial'].fill(new_values)


###############
# RUN
###############

if __name__ == '__main__':
    main()
