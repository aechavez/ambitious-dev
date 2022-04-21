import numpy as np
import os
import ROOT as r
from modules import mipTracking, physTools
from modules import rootManager as manager

cell_map = np.loadtxt('modules/cellModule.txt')
r.gSystem.Load('libFramework.so')

# Branch information to build tree models
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

# Main subroutine
def main():

    # Parse the arguments passed by the user
    parsing_dict = manager.parse()
    batch_mode = parsing_dict['batch_mode']
    separate_categories = parsing_dict['separate_categories']
    inputs = parsing_dict['inputs']
    group_labels = parsing_dict['group_labels']
    outputs = parsing_dict['outputs']
    start_event = parsing_dict['start_event']
    max_events = parsing_dict['max_events']

    # Initialize each process
    processes = []
    for label, group in zip(group_labels, inputs):
        processes.append(manager.TreeProcess(process_event, file_group = group, name_tag = label,\
                                             start_event = start_event, max_events = max_events,\
                                             batch_mode = batch_mode))

    # Run each process
    for process in processes:

        # Move into the appropriate temporary directory
        os.chdir(process.temporary_directory)

        # Add branches needed for BDT analysis
        process.target_sp_hits = process.add_branch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        process.ecal_sp_hits = process.add_branch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        process.ecal_rec_hits = process.add_branch('EcalHit', 'EcalRecHits_v12')
        process.ecal_veto = process.add_branch('EcalVetoResult', 'EcalVeto_v12')

        process.separate_categories = separate_categories

        if process.separate_categories:
            process.tree_models = {
                'fiducial_electron_photon': None,
                'fiducial_electron': None,
                'fiducial_photon': None,
                'non_fiducial': None
            }

        else:
            process.tree_models = {'unsorted': None}

        for tree_model in process.tree_models:
            process.tree_models[tree_model] = manager.TreeMaker('{}_{}.root'.format(group_labels[processes.index(process)], tree_model),\
                                                                'EcalVeto', branch_information = branch_information,\
                                                                out_directory = outputs[processes.index(process)])

        # Set closing functions
        process.closing_functions = [process.tree_models[tree_model].write\
                                     for tree_model in process.tree_models]

        # Run this process
        process.run()

    # Remove the scratch directory if there is one
    # (Being careful not to break other jobs if we are in batch mode)
    if not batch_mode:
        manager.remove_scratch()

    print('\n[ INFO ] - All processes done!')

# Subroutine to process an event
def process_event(self):

    # Reset the branch values for each tree model
    for tree_model in self.tree_models:
        self.tree_models[tree_model].reset_values()

    # Initialize a dictionary of new values
    new_values = {branch_name: branch_information[branch_name]['default'] for branch_name in branch_information}


    ###########################################
    # Electron and photon information
    ###########################################

    # Get the electron's position and momentum at the target
    ele_target_sp_hit = physTools.get_electron_target_sp_hit(self.target_sp_hits)

    if not (ele_target_sp_hit is None):
        ele_target_sp_pos = physTools.get_position(ele_target_sp_hit)
        ele_target_sp_mom = physTools.get_momentum(ele_target_sp_hit)

    else:
        print('[ WARNING ] - No electron found at the target!')
        ele_target_sp_pos = ele_target_sp_mom = np.zeros(3)

    # Get the electron's position and momentum at the ECal
    ele_ecal_sp_hit = physTools.get_electron_ecal_sp_hit(self.ecal_sp_hits)

    if not (ele_ecal_sp_hit is None):
        ele_ecal_sp_pos = physTools.get_position(ele_ecal_sp_hit)
        ele_ecal_sp_mom = physTools.get_momentum(ele_ecal_sp_hit)

    else:
        print('[ WARNING ] - No electron found at the ECal!')
        ele_ecal_sp_pos = ele_ecal_sp_mom = np.zeros(3)

    # Infer the photon's position and momentum at the target
    if not (ele_target_sp_hit is None):
        pho_target_sp_pos, pho_target_sp_mom = physTools.infer_photon_target_sp_hit(ele_target_sp_hit)

    else:
        pho_target_sp_pos = pho_target_sp_mom = np.zeros(3)

    # Use linear projections to infer the electron and photon trajectories
    ele_traj = pho_traj = None

    if not (ele_ecal_sp_hit is None):
        ele_traj = physTools.intercepts(ele_ecal_sp_pos, ele_ecal_sp_mom, physTools.ecal_layerZs)

    if not (ele_target_sp_hit is None):
        pho_traj = physTools.intercepts(pho_target_sp_pos, pho_target_sp_mom, physTools.ecal_layerZs)

    # If desired, determine which fiducial category the event belongs to
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

    new_values['nReadoutHits']    = self.ecal_veto.getNReadoutHits()
    new_values['summedDet']       = self.ecal_veto.getSummedDet()
    new_values['summedTightIso']  = self.ecal_veto.getSummedTightIso()
    new_values['maxCellDep']      = self.ecal_veto.getMaxCellDep()
    new_values['showerRMS']       = self.ecal_veto.getShowerRMS()
    new_values['xStd']            = self.ecal_veto.getXStd()
    new_values['yStd']            = self.ecal_veto.getYStd()
    new_values['avgLayerHit']     = self.ecal_veto.getAvgLayerHit()
    new_values['stdLayerHit']     = self.ecal_veto.getStdLayerHit()
    new_values['deepestLayerHit'] = self.ecal_veto.getDeepestLayerHit() 
    new_values['ecalBackEnergy']  = self.ecal_veto.getEcalBackEnergy()


    ############################################
    # Calculate MIP tracking variables
    ############################################

    # Calculate trajectorySeparation and trajectoryDot
    if not ((ele_traj is None) and (pho_traj is None)):

        # Arrays marking start/endpoints of each trajectory
        ele_traj_ends = np.array([[ele_traj[0][0], ele_traj[0][1], physTools.ecal_layerZs[0]],\
                                  [ele_traj[-1][0], ele_traj[-1][1], physTools.ecal_layerZs[-1]]])
        pho_traj_ends = np.array([[pho_traj[0][0], pho_traj[0][1], physTools.ecal_layerZs[0]],\
                                  [pho_traj[-1][0], pho_traj[-1][1], physTools.ecal_layerZs[-1]]])

        ele_traj_vec = physTools.normalize(ele_traj_ends[1] - ele_traj_ends[0])
        pho_traj_vec = physTools.normalize(pho_traj_ends[1] - pho_traj_ends[0])

        new_values['trajectorySeparation'] = physTools.distance(ele_traj_ends[0], pho_traj_ends[0])
        new_values['trajectoryDot'] = np.dot(ele_traj_vec, pho_traj_vec)

    else:

        # One of the trajectories is missing, so use all of the hits in the ECal for MIP tracking
        # Take endpoints far outside the ECal so they don't restrict tracking
        ele_traj_ends = np.array([[999., 999., 0.], [999., 999., 999.]])
        pho_traj_ends = np.array([[1000., 1000., 0.], [1000., 1000., 1000.]])

        # Assign dummy values in this case
        new_values['trajectorySeparation'] = 11.
        new_values['trajectoryDot'] = 4.

    # Setup for territory variables
    pho_to_ele = physTools.normalize(ele_traj_ends[0] - pho_traj_ends[0])
    origin = 0.5*physTools.cell_width*pho_to_ele + pho_traj_ends[0]


    #########################################
    # Radius of containment binning
    #########################################

    # Recoil momentum magnitude and angle with respect to Z-axis
    recoil_mom_mag = recoil_mom_theta = -1.

    if not (ele_ecal_sp_hit is None):
        recoil_mom_mag = np.linalg.norm(ele_ecal_sp_mom)
        recoil_mom_theta = physTools.angle(ele_ecal_sp_mom, np.array([0., 0., 1.]), units = 'degrees')

    # Set electron RoC binnings
    ele_radii = physTools.radius68_thetalt10_plt500

    if recoil_mom_theta < 10. and recoil_mom_mag >= 500.:
        ele_radii = physTools.radius68_thetalt10_pgt500

    elif recoil_mom_theta >= 10. and recoil_mom_theta < 20.:
        ele_radii = physTools.radius68_theta10to20

    elif recoil_mom_theta >= 20.:
        ele_radii = physTools.radius68_thetagt20

    # Use default binning for photon RoC
    pho_radii = physTools.radius68_thetalt10_plt500


    ###########################
    # Major ECal loop
    ###########################

    # List of hits for MIP tracking
    tracking_hit_list = []

    for hit in self.ecal_rec_hits:
        
        if hit.getEnergy() > 0:

            layer = physTools.ecal_layer(hit)
            xy_pair = ( hit.getXPos(), hit.getYPos() )

            # Territory selections
            hitPrime = physTools.pos(hit) - origin
            if np.dot(hitPrime, gToe) > 0: feats['fullElectronTerritoryHits'] += 1
            else: feats['fullPhotonTerritoryHits'] += 1

            # Distance to electron trajectory
            if e_traj != None:
                xy_e_traj = ( e_traj[layer][0], e_traj[layer][1] )
                distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
            else: distance_e_traj = -1.0

            # Distance to photon trajectory
            if g_traj != None:
                xy_g_traj = ( g_traj[layer][0], g_traj[layer][1] )
                distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
            else: distance_g_traj = -1.0

            # Decide which longitudinal segment the hit is in and add to sums
            for i in range(1, physTools.nSegments + 1):

                if (physTools.segLayers[i - 1] <= layer)\
                  and (layer <= physTools.segLayers[i] - 1):
                    feats['energy_s{}'.format(i)] += hit.getEnergy()
                    feats['nHits_s{}'.format(i)] += 1
                    feats['xMean_s{}'.format(i)] += xy_pair[0]*hit.getEnergy()
                    feats['yMean_s{}'.format(i)] += xy_pair[1]*hit.getEnergy()
                    feats['layerMean_s{}'.format(i)] += layer*hit.getEnergy()

                    # Decide which containment region the hit is in and add to sums
                    for j in range(1, physTools.nRegions + 1):

                        if ((j - 1)*e_radii[layer] <= distance_e_traj)\
                          and (distance_e_traj < j*e_radii[layer]):
                            feats['eContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['eContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['eContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['eContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['eContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

                        if ((j - 1)*g_radii[layer] <= distance_g_traj)\
                          and (distance_g_traj < j*g_radii[layer]):
                            feats['gContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['gContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['gContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['gContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['gContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

                        if (distance_e_traj > j*e_radii[layer])\
                          and (distance_g_traj > j*g_radii[layer]):
                            feats['oContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['oContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['oContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['oContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['oContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

            # Build MIP tracking hit list; (outside electron region or electron missing)
            if distance_e_traj >= e_radii[layer] or distance_e_traj == -1.0:
                trackingHitList.append(hit) 

    # If possible, quotient out the total energy from the means
    for i in range(1, physTools.nSegments + 1):

        if feats['energy_s{}'.format(i)] > 0:
            feats['xMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]
            feats['yMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]
            feats['layerMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]

        for j in range(1, physTools.nRegions + 1):

            if feats['eContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['eContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]
                feats['eContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]
                feats['eContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]

            if feats['gContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['gContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]
                feats['gContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]
                feats['gContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]

            if feats['oContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['oContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]
                feats['oContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]
                feats['oContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]

    # Loop over hits again to calculate the standard deviations
    for hit in self.ecalRecHits:

        layer = physTools.ecal_layer(hit)
        xy_pair = (hit.getXPos(), hit.getYPos())

        # Distance to electron trajectory
        if e_traj != None:
            xy_e_traj = (e_traj[layer][0], e_traj[layer][1])
            distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
        else:
            distance_e_traj = -1.0

        # Distance to photon trajectory
        if g_traj != None:
            xy_g_traj = (g_traj[layer][0], g_traj[layer][1])
            distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
        else:
            distance_g_traj = -1.0

        # Decide which longitudinal segment the hit is in and add to sums
        for i in range(1, physTools.nSegments + 1):

            if (physTools.segLayers[i - 1] <= layer) and\
                    (layer <= physTools.segLayers[i] - 1):
                feats['xStd_s{}'.format(i)] += ((xy_pair[0] -\
                        feats['xMean_s{}'.format(i)])**2)*hit.getEnergy()
                feats['yStd_s{}'.format(i)] += ((xy_pair[1] -\
                        feats['yMean_s{}'.format(i)])**2)*hit.getEnergy()
                feats['layerStd_s{}'.format(i)] += ((layer -\
                        feats['layerMean_s{}'.format(i)])**2)*hit.getEnergy()

                # Decide which containment region the hit is in and add to sums
                for j in range(1, physTools.nRegions + 1):

                    if ((j - 1)*e_radii[layer] <= distance_e_traj)\
                      and (distance_e_traj < j*e_radii[layer]):
                        feats['eContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['eContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['eContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['eContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['eContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['eContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()

                    if ((j - 1)*g_radii[layer] <= distance_g_traj)\
                      and (distance_g_traj < j*g_radii[layer]):
                        feats['gContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['gContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['gContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['gContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['gContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['gContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()

                    if (distance_e_traj > j*e_radii[layer])\
                      and (distance_g_traj > j*g_radii[layer]):
                        feats['oContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['oContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['oContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['oContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['oContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['oContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()

    # Quotient out the total energies from the standard deviations if possible and take root
    for i in range(1, physTools.nSegments + 1):

        if feats['energy_s{}'.format(i)] > 0:
            feats['xStd_s{}'.format(i)] = math.sqrt(feats['xStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])
            feats['yStd_s{}'.format(i)] = math.sqrt(feats['yStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])
            feats['layerStd_s{}'.format(i)] = math.sqrt(feats['layerStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])

        for j in range(1, physTools.nRegions + 1):

            if feats['eContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['eContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContXStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])
                feats['eContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContYStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])
                feats['eContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])

            if feats['gContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['gContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContXStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])
                feats['gContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContYStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])
                feats['gContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])

            if feats['oContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['oContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContXStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])
                feats['oContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContYStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])
                feats['oContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])


    # Find the first layer of the ECal where a hit near the projected photon trajectory
    # AND the total number of hits around the photon trajectory
    if g_traj != None: # If no photon trajectory, leave this at the default

        # First currently unusued; pending further study; performance drop from  v9 and v12
        #print(trackingHitList, g_traj)
        feats['firstNearPhLayer'], feats['nNearPhHits'] = mipTracking.nearPhotonInfo(
                                                            trackingHitList, g_traj )
    else: feats['nNearPhHits'] = feats['nReadoutHits']


    # Territories limited to trackingHitList
    if e_traj != None:
        for hit in trackingHitList:
            hitPrime = physTools.pos(hit) - origin
            if np.dot(hitPrime, gToe) > 0: feats['electronTerritoryHits'] += 1
            else: feats['photonTerritoryHits'] += 1
    else:
        feats['photonTerritoryHits'] = feats['nReadoutHits']
        feats['TerritoryRatio'] = 10
        feats['fullTerritoryRatio'] = 10
    if feats['electronTerritoryHits'] != 0:
        feats['TerritoryRatio'] = feats['photonTerritoryHits']/feats['electronTerritoryHits']
    if feats['fullElectronTerritoryHits'] != 0:
        feats['fullTerritoryRatio'] = feats['fullPhotonTerritoryHits']/\
                                            feats['fullElectronTerritoryHits']


    # Find MIP tracks
    feats['straight4'], trackingHitList = mipTracking.findStraightTracks(
                                trackingHitList, e_traj_ends, g_traj_ends,
                                mst = 4, returnHitList = True)

    # Fill the tree (according to fiducial category) with values for this event
    if not self.separate:
        self.tfMakers['unsorted'].fillEvent(feats)
    else:
        if e_fid and g_fid: self.tfMakers['egin'].fillEvent(feats)
        elif e_fid and not g_fid: self.tfMakers['ein'].fillEvent(feats)
        elif not e_fid and g_fid: self.tfMakers['gin'].fillEvent(feats)
        else: self.tfMakers['none'].fillEvent(feats)

if __name__ == "__main__":
    main()
