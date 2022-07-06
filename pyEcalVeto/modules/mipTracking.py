import numpy as np
from modules import physTools


# NOTE: Don't forget to order hits by reverse zpos before using the nXTracks funcs
# This is assumed to make use of some algorithm short cuts

##########################
# First layer with hit within one cell of the photon trajectory AND
# Number of hits within one cell of the photon trajectory
##########################
def nearPhotonInfo(trackingHitList, g_trajectory, returnLayer=True, returnNumber=True):

    layer = 33
    n = 0
    for hit in trackingHitList:

        # Near the photn trajectory
        if physTools.distance( physTools.get_position(hit)[:2],
                g_trajectory[ physTools.get_ecal_layer( hit ) ] ) < physTools.cell_width:
            n += 1

            # Earliest layer
            if physTools.get_ecal_layer( hit ) < layer:
                layer = physTools.get_ecal_layer( hit )

    # Prepare and return desired output
    out = []
    if returnLayer: out.append(layer)
    if returnNumber: out.append(n)

    return out

##########################
# Straight tracks
##########################

# Based on previous python; All of v9 analysis done with this
def findStraightTracks(hitlist, etraj_ends, ptraj_ends,\
                        mst = 2, returnN=True, returnHitList = False, returnTracks = False):

    strtracklist = []   # Initialize output
    hitscopy = hitlist  # Need this because hitlist gets eddited

    for hit in hitlist:  #Go through all hits, starting at the back of the ecal
        track = [hit]
        currenthit = hit  #Stores "trailing" hit in track being constructed
        possibleNeigh = False
        for h in hitscopy:
            if h.getZPos() == currenthit.getZPos():
                possibleNeigh = True  #Optimization
                continue
            if not possibleNeigh:  continue
            if currenthit.getZPos() - h.getZPos() > 25:  #Optimization
                possibleNeigh = False
                continue
            neighFound = (
                    (physTools.get_ecal_layer(h) ==\
                            physTools.get_ecal_layer(currenthit) - 1 or\
                     physTools.get_ecal_layer(h) ==\
                            physTools.get_ecal_layer(currenthit) -2) and\
                     h.getXPos() == currenthit.getXPos() and\
                     h.getYPos() == currenthit.getYPos() )
            if neighFound:
                track.append(h)
                currenthit = h

        # Too short
        if len(track) < mst: continue

        # If it's exactly the min, it has to be very close to ptraj
        if len(track) == mst: 
            for hitt in track:
                if physTools.point_line_dist( physTools.get_position(hitt),
                        ptraj_ends[0], ptraj_ends[1] ) > physTools.cell_width - 0.5:
                    break
                continue

        # Check that the track approaches the photon's and not the electron's
        trk_s = np.array( (track[ 0].getXPos(), track[ 0].getYPos(), track[ 0].getZPos() ) )
        trk_e = np.array( (track[-1].getXPos(), track[-1].getYPos(), track[-1].getZPos() ) )
        closest_e = physTools.line_line_dist( etraj_ends[0], etraj_ends[1], trk_s, trk_e )
        closest_p = physTools.line_line_dist( ptraj_ends[0], ptraj_ends[1], trk_s, trk_e )
        if closest_p > physTools.cell_width and closest_e < 2*physTools.cell_width:
            continue

        # Remove hits in current track from further consideration
        for h in track:
            hitlist.remove(h)

        # Append track to track list
        strtracklist.append(track)

    # Combine tracks that should be consecutive
    # NOTE: Should maybe do this eariler in case 2 len=2 tracks add up to a passing 4
    strtracklist.sort(key=lambda h: hit.getZPos(), reverse=True) # Should be done check this

    currentInd = 0
    while currentInd < len(strtracklist):

        trk = strtracklist[currentInd]
        tmpInd = currentInd+1
        mergeFound = False

        # Search for track compatible with current one
        while tmpInd < len(strtracklist) and not mergeFound:
            trk_ = strtracklist[tmpInd]
            trk_e = np.array( (track[-1].getXPos(), track[-1].getYPos(),
                                                    track[-1].getZPos() ) )
            trk_s = np.array( (track[ 0].getXPos(), track[ 0].getYPos(),
                                                    track[ 0].getZPos() ) )
            # If head+tail are w/in one cell of each other
            if physTools.distance( trk_e, trk_s ) < physTools.cell_width:
                for hit in trk_:
                    trk.append(hit)
                strtracklist.remove(trk_)
                mergeFound = True
            tmpInd += 1
        if not mergeFound:
            currentInd += 1

    # Prepare and return desired output
    out = []
    if returnN: out.append( len(strtracklist) )
    if returnHitList: out.append( hitlist )
    if returnTracks: out.append( strtracklist )

    return out
