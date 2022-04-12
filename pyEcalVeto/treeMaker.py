import os
import math
import ROOT as r
import numpy as np
from modules import physTools, mipTracking
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
    'firstNearPhLayer':          {'dtype': int,   'default': 33},
    'nNearPhHits':               {'dtype': int,   'default': 0 },
    'fullElectronTerritoryHits': {'dtype': int,   'default': 0 },
    'fullPhotonTerritoryHits':   {'dtype': int,   'default': 0 },
    'fullTerritoryRatio':        {'dtype': float, 'default': 1.},
    'electronTerritoryHits':     {'dtype': int,   'default': 0 },
    'photonTerritoryHits':       {'dtype': int,   'default': 0 },
    'TerritoryRatio':            {'dtype': float, 'default': 1.},
    'epSep':                     {'dtype': float, 'default': 0.},
    'epDot':                     {'dtype': float, 'default': 0.}

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
        branch_information['eContEnergy_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}
        branch_information['eContNHits_x{}_s{}'.format(j, i)]     = {'dtype': int,   'default': 0 }
        branch_information['eContXMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['eContYMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['eContLayerMean_x{}_s{}'.format(j, i)] = {'dtype': float, 'default': 0.}
        branch_information['eContXStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['eContYStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['eContLayerStd_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}

        # Photon RoC variables
        branch_information['gContEnergy_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}
        branch_information['gContNHits_x{}_s{}'.format(j, i)]     = {'dtype': int,   'default': 0 }
        branch_information['gContXMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['gContYMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['gContLayerMean_x{}_s{}'.format(j, i)] = {'dtype': float, 'default': 0.}
        branch_information['gContXStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['gContYStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['gContLayerStd_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}

        # Outside RoC variables
        branch_information['oContEnergy_x{}_s{}'.format(j, i)]    = {'dtype': float, 'default': 0.}
        branch_information['oContNHits_x{}_s{}'.format(j, i)]     = {'dtype': int,   'default': 0 }
        branch_information['oContXMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['oContYMean_x{}_s{}'.format(j, i)]     = {'dtype': float, 'default': 0.}
        branch_information['oContLayerMean_x{}_s{}'.format(j, i)] = {'dtype': float, 'default': 0.}
        branch_information['oContXStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['oContYStd_x{}_s{}'.format(j, i)]      = {'dtype': float, 'default': 0.}
        branch_information['oContLayerStd_x{}_s{}'.format(j, i)]  = {'dtype': float, 'default': 0.}

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

    # Build each process
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
        process.ecal_veto = process.add_branch('EcalVetoResult', 'EcalVeto_v12')
        process.target_sp_hits = process.add_branch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        process.ecal_sp_hits = process.add_branch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        process.ecal_rec_hits = process.add_branch('EcalHit', 'EcalRecHits_v12')

        process.separate_categories = separate_categories

        if process.separate_categories:
            process.tree_models = {
                'electron_photon_fiducial': None,
                'electron_fiducial': None,
                'photon_fiducial': None,
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
    # (Just grab from the first tree model, since they're identical)
    new_values = self.tree_models[next(iter(self.tree_models))].branches

    # Assign pre-computed variables
    feats['nReadoutHits']       = self.ecalVeto.getNReadoutHits()
    feats['summedDet']          = self.ecalVeto.getSummedDet()
    feats['summedTightIso']     = self.ecalVeto.getSummedTightIso()
    feats['maxCellDep']         = self.ecalVeto.getMaxCellDep()
    feats['showerRMS']          = self.ecalVeto.getShowerRMS()
    feats['xStd']               = self.ecalVeto.getXStd()
    feats['yStd']               = self.ecalVeto.getYStd()
    feats['avgLayerHit']        = self.ecalVeto.getAvgLayerHit()
    feats['stdLayerHit']        = self.ecalVeto.getStdLayerHit()
    feats['deepestLayerHit']    = self.ecalVeto.getDeepestLayerHit() 
    feats['ecalBackEnergy']     = self.ecalVeto.getEcalBackEnergy()
    
    ###################################
    # Determine event type
    ###################################

    # Get e position and momentum from EcalSP
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    if e_ecalHit != None:
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()

    # Photon Info from targetSP
    e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
    if e_targetHit != None:
        g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
    else:  # Should about never happen -> division by 0 in g_traj
        print('no e at targ!')
        g_targPos = g_targP = np.zeros(3)

    # Get electron and photon trajectories
    e_traj = g_traj = None

    if e_ecalHit != None:
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)

    if e_targetHit != None:
        g_traj = physTools.layerIntercepts(g_targPos, g_targP)

    # Fiducial categories (filtered into different output trees)
    if self.separate:
        e_fid = g_fid = False

        if e_traj != None:
            for cell in cellMap:
                if physTools.dist( cell[1:], e_traj[0] ) <= physTools.cell_radius:
                    e_fid = True
                    break

        if g_traj != None:
            for cell in cellMap:
                if physTools.dist( cell[1:], g_traj[0] ) <= physTools.cell_radius:
                    g_fid = True
                    break

    ###################################
    # Compute extra BDT input variables
    ###################################

    # Find epSep and epDot, and prepare electron and photon trajectory vectors
    if e_traj != None and g_traj != None:

        # Create arrays marking start and end of each trajectory
        e_traj_ends = [np.array([e_traj[0][0], e_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([e_traj[-1][0], e_traj[-1][1], physTools.ecal_layerZs[-1] ])]
        g_traj_ends = [np.array([g_traj[0][0], g_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([g_traj[-1][0], g_traj[-1][1], physTools.ecal_layerZs[-1] ])]

        # Unused epDot and epSep
        #e_norm  = physTools.unit( e_traj_ends[1] - e_traj_ends[0] )
        #g_norm  = physTools.unit( g_traj_ends[1] - g_traj_ends[0] )
        #feats['epSep'] = physTools.dist( e_traj_ends[0], g_traj_ends[0] )
        #feats['epDot'] = physTools.dot(e_norm,g_norm)

    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_ends   = [np.array([999 ,999 ,0   ]), np.array([999 ,999 ,999 ]) ]
        g_traj_ends   = [np.array([1000,1000,0   ]), np.array([1000,1000,1000]) ]

        #feats['epSep'] = 10.0 + 1.0 # Don't cut on these in this case
        #feats['epDot'] = 3.0 + 1.0

    # Territory setup (consider missing case)
    gToe    = physTools.unit( e_traj_ends[0] - g_traj_ends[0] )
    origin  = g_traj_ends[0] + 0.5*8.7*gToe

    # Recoil electron momentum magnitude and angle with z-axis
    recoilPMag  = physTools.mag(  e_ecalP )                 if e_ecalHit != None else -1.0
    recoilTheta = physTools.angle(e_ecalP, units='radians') if recoilPMag > 0    else -1.0

    # Set electron RoC binnings
    e_radii = physTools.radius68_thetalt10_plt500
    if recoilTheta < 10 and recoilPMag >= 500: e_radii = physTools.radius68_thetalt10_pgt500
    elif recoilTheta >= 10 and recoilTheta < 20: e_radii = physTools.radius68_theta10to20
    elif recoilTheta >= 20: e_radii = physTools.radius68_thetagt20

    # Always use default binning for photon RoC
    g_radii = physTools.radius68_thetalt10_plt500

    # Big data
    trackingHitList = []

    # Major ECal loop
    for hit in self.ecalRecHits:
        
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
