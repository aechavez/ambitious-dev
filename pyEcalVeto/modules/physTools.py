import numpy as np


#############################
# General constants
#############################

# Constant defining the clearance between volumes
clearance = 0.001

# Thickness of the scoring planes
sp_thickness = 0.001


############################
# Target constants
############################

# Position
target_z = 0.

# Tungsten X0 = 0.3504cm
# Target thickness = 0.1X0
target_thickness = 0.3504

# Target dimensions
target_dim_x = 40.
target_dim_y = 100.


#################################
# Target scoring planes
#################################

# Surround the target with scoring planes
sp_target_down_z = target_z + target_thickness/2 + sp_thickness/2 + clearance
sp_target_up_z = target_z - target_thickness/2 - sp_thickness/2 - clearance


##########################################
# Trigger scintillator constants
##########################################

# Trigger scintillator positions
trigger_pad_thickness = 4.5
trigger_pad_bar_thickness = 2.
trigger_pad_bar_gap = 0.3
trigger_pad_dim_x = target_dim_x
trigger_pad_dim_y = target_dim_y
trigger_bar_dx = 40.
trigger_bar_dy = 3.
number_of_bars = 25

trigger_pad_offset = (target_dim_y - (number_of_bars*trigger_bar_dy + (number_of_bars - 1)*trigger_pad_bar_gap))/2

# Trigger pad distance from the target is -2.4262mm
trigger_pad_up_z = target_z - (target_thickness/2) - (trigger_pad_thickness/2) - clearance

# Trigger pad distance from the target is 2.4262mm
trigger_pad_down_z = target_z + (target_thickness/2) + (trigger_pad_thickness/2) + clearance


###############################################
# Trigger scintillator scoring planes
###############################################

# Place scoring planes downstream of each trigger scintillator array
sp_trigger_pad_down_l1_z = trigger_pad_down_z - trigger_pad_bar_gap/2 + sp_thickness/2 + clearance
sp_trigger_pad_down_l2_z = trigger_pad_down_z + trigger_pad_bar_gap/2 + trigger_pad_bar_thickness + sp_thickness/2 + clearance
sp_trigger_pad_up_l1_z = trigger_pad_up_z - trigger_pad_bar_gap/2 + sp_thickness/2 + clearance
sp_trigger_pad_up_l2_z = trigger_pad_up_z + trigger_pad_bar_gap/2 + trigger_pad_bar_thickness + sp_thickness/2 + clearance


##########################
# ECal constants
##########################

# ECal layer thicknesses
Wthick_A_dz = 0.75
W_A_dz = 0.75
Wthick_B_dz = 2.25
W_B_dz = 1.5
Wthick_C_dz = 3.5
W_C_dz = 1.75
Wthick_D_dz = 7.
W_D_dz = 3.5
CFMix_dz = 0.05
CFMixThick_dz = 0.2
PCB_dz = 1.5
Si_dz = 0.5
C_dz = 0.5
Al_dz = 2.

# Air separating sheets of Al or W with PCB motherboard
# Limited by construction abilities 
FrontTolerance = 0.5

# Gap between layers
BackTolerance = 0.5

# Air separating PCBs from PCB motherboards
PCB_Motherboard_Gap = 2.3

# Air separating Carbon sheets in the middle of a layer
CoolingAirGap = 4.

# Preshower thickness is 20.1mm
preshower_Thickness = Al_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                      + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + CoolingAirGap\
                      + 2*C_dz + CFMixThick_dz + Si_dz + CFMix_dz + PCB_dz\
                      + PCB_Motherboard_Gap + PCB_dz + BackTolerance

# Layer A thickness is 20.35mm
layer_A_Thickness = Wthick_A_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_A_dz + C_dz\
                    + CoolingAirGap + C_dz + W_A_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# GDML comment indicates Layer B thickness is 22.35mm, but the actual value is 23.35mm!
layer_B_Thickness = Wthick_B_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_B_dz + C_dz\
                    + CoolingAirGap + C_dz + W_B_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# Layer C thickness is 25.1mm
layer_C_Thickness = Wthick_C_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_C_dz + C_dz\
                    + CoolingAirGap + C_dz + W_C_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# Layer D thickness is 32.1mm
layer_D_Thickness = Wthick_D_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_D_dz + C_dz\
                    + CoolingAirGap + C_dz + W_D_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# Number of layers
ecal_A_layers = 1
ecal_B_layers = 1
ecal_C_layers = 9
ecal_D_layers = 5

# GDML comment indicates ECal thickness is 449.2mm,
# but the actual value is 450.2mm!
ECal_dz = preshower_Thickness\
          + layer_A_Thickness*ecal_A_layers\
          + layer_B_Thickness*ecal_B_layers\
          + layer_C_Thickness*ecal_C_layers\
          + layer_D_Thickness*ecal_D_layers

# Flat-to-flat gap between modules
module_gap = 1.5

# Center-to-flat radius of one module
module_radius = 85.

module_side = (2/np.sqrt(3))*module_radius

# ECal width and height
ECal_dx = (10/np.sqrt(3))*module_radius + np.sqrt(3)*module_gap
ECal_dy = 6*module_radius + 2*module_gap

# GDML comment indicates distance from target to ECal is 220.5mm,
# but the actual value is clearly 240.5mm!
ecal_front_z = 240.

side_Ecal_dx = 800.
side_Ecal_dy = 600.

# Dimensions of ECal parent volume. The size is set to be 1mm larger
# than the thickness of the ECal calculated above
ecal_envelope_x = side_Ecal_dx
ecal_envelope_y = side_Ecal_dy
ecal_envelope_z = ECal_dz + 1.


###############################
# ECal scoring planes
###############################

# Surround the ECal with scoring planes
sp_ecal_front_z = ecal_front_z + (ecal_envelope_z - ECal_dz)/2 - sp_thickness/2 + clearance
sp_ecal_back_z = ecal_front_z + ECal_dz + (ecal_envelope_z - ECal_dz)/2 + sp_thickness/2
sp_ecal_top_y = ECal_dy/2 + sp_thickness/2
sp_ecal_bot_y = -ECal_dy/2 - sp_thickness/2
sp_ecal_left_x = -ECal_dx/2 - sp_thickness/2
sp_ecal_right_x = ECal_dx/2 + sp_thickness/2
sp_ecal_mid_z = ecal_front_z + ECal_dz/2 + (ecal_envelope_z - ECal_dz)/2


#####################################
# ECal detector description
#####################################

ecal_layerZs = ecal_front_z  + (ecal_envelope_z - ECal_dz)/2 + np.array([7.850,   13.300,  26.400,  33.500,  47.950,
                                                                         56.550,  72.250,  81.350,  97.050,  106.150,
                                                                         121.850, 130.950, 146.650, 155.750, 171.450,
                                                                         180.550, 196.250, 205.350, 221.050, 230.150,
                                                                         245.850, 254.950, 270.650, 279.750, 298.950,
                                                                         311.550, 330.750, 343.350, 362.550, 375.150,
                                                                         394.350, 406.950, 426.150, 438.750])

nCellRHeight = 35.3

# Center-to-corner radius of one cell
cell_radius = (2/nCellRHeight)*module_radius

# Center-to-flat diameter of one cell
cell_width = np.sqrt(3)*cell_radius

# Space for up to 64 layers
ecal_LAYER_MASK = 0x3F
ecal_LAYER_SHIFT = 17

# Space for up to 32 modules/layer
ecal_MODULE_MASK = 0x1F
ecal_MODULE_SHIFT = 12

# Space for up to 4096 cells/module
ecal_CELL_MASK = 0xFFF
ecal_CELL_SHIFT = 0


##########################
# HCal constants
##########################

# Width and height of the envelope for the side and back HCal 
# Must be the maximum of back HCal dx and side HCal dx 
hcal_envelope_dx = 3100.
hcal_envelope_dy = 3100.

# Common HCal components
air_thick = 2.
scint_thick = 20.

# Back HCal layer component
# Layer 1 has no absorber, layers 2 and 3 have absorber of different thickness
hcal_back_dx = 3100.
hcal_back_dy = 3100.
back_numLayers1 = 0
back_numLayers2 = 100
back_numLayers3 = 0
back_abso2_thick = 25.
back_abso3_thick = 50.
back_layer1_thick = scint_thick + air_thick
back_layer2_thick = back_abso2_thick + scint_thick + 2*air_thick
back_layer3_thick = back_abso3_thick + scint_thick + 2*air_thick
hcal_back_dz1 = back_numLayers1*back_layer1_thick
hcal_back_dz2 = back_numLayers2*back_layer2_thick
hcal_back_dz3 = back_numLayers3*back_layer3_thick
hcal_back_dz = hcal_back_dz1 + hcal_back_dz2 + hcal_back_dz3

# Side HCal Layer component
sideTB_layers = 28
sideLR_layers = 26
side_abso_thick = 20.

# Side dz has to be greater than side ECal dz
hcal_side_dz = 600.

# Total calorimeter thickness
hcal_dz = hcal_back_dz + hcal_side_dz


###############################
# HCal scoring planes
###############################

# Surround the HCal with scoring planes
sp_hcal_front_z = ecal_front_z - sp_thickness/2 + clearance
sp_hcal_back_z = ecal_front_z + hcal_back_dz + hcal_side_dz + sp_thickness/2
sp_hcal_top_y = hcal_back_dy/2 + sp_thickness/2
sp_hcal_bot_y = -hcal_back_dy/2 - sp_thickness/2
sp_hcal_left_x = -hcal_back_dx/2 - sp_thickness/2
sp_hcal_right_x = hcal_back_dx/2 + sp_thickness/2
sp_hcal_mid_z = ecal_front_z + hcal_dz/2


#####################################
# HCal detector description
#####################################

# Space for up to 7 sections
hcal_SECTION_MASK = 0x7
hcal_SECTION_SHIFT = 18

# Space for up to 255 layers
hcal_LAYER_MASK = 0xFF
hcal_LAYER_SHIFT = 10

# Space for up to 255 strips/layer
hcal_STRIP_MASK = 0xFF
hcal_STRIP_SHIFT = 0


###################################
# Miscellaneous constants
###################################

# Arrays holding 68% containment radii/layer for different bins in momentum/angle
radius68_thetalt10_plt500 = np.array([4.045666158618167,  4.086393662224346,  4.359141107602775,  4.666549994726691,  5.8569181911416015,
                                      6.559716356124256,  8.686967529043072,  10.063482736354674, 13.053528344041274, 14.883496407943747,
                                      18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406,  28.329169392203216,
                                      29.468032123356345, 34.03271241527079,  35.03747443690781,  38.50748727211848,  39.41576583301171,
                                      42.63622296033334,  45.41123601592071,  48.618139095742876, 48.11801717451056,  53.220539860213655,
                                      58.87753380915155,  66.31550881539764,  72.94685877928593,  85.95506228335348,  89.20607201266672,
                                      93.34370253818409,  96.59471226749734,  100.7323427930147,  103.98335252232795])
radius68_thetalt10_pgt500 = np.array([4.081926458777424,  4.099431732299409,  4.262428482867968,  4.362017581473145,  4.831341579961153,
                                      4.998346041276382,  6.2633736512415705, 6.588371889265881,  8.359969947444522,  9.015085558044309,
                                      11.262722588206483, 12.250305471269183, 15.00547660437276,  16.187264014640103, 19.573764900578503,
                                      20.68072032434797,  24.13797140783321,  25.62942209291236,  29.027596514735617, 30.215039667389316,
                                      33.929540248019585, 36.12911729771914,  39.184563500620946, 42.02062468386282,  46.972125628650204,
                                      47.78214816041894,  55.88428562462974,  59.15520134927332,  63.31816666637158,  66.58908239101515,
                                      70.75204770811342,  74.022963432757,    78.18592874985525,  81.45684447449884])
radius68_theta10to20 = np.array([4.0251896715647115, 4.071661598616328, 4.357690094817289,  4.760224640141712,  6.002480766325418,
                                 6.667318981016246,  8.652513285172342, 9.72379373302137,   12.479492693251478, 14.058548828317289,
                                 17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302,
                                 31.768946746958296, 35.79247330197688, 37.27810357669942,  41.657281051476545, 42.628141392692626,
                                 47.94208483539388,  49.9289473559796,  54.604030254423975, 53.958762417361655, 53.03339560920388,
                                 57.026277390001425, 62.10810455035879, 66.10098633115634,  71.1828134915137,   75.17569527231124,
                                 80.25752243266861,  84.25040421346615, 89.33223137382352,  93.32511315462106])
radius68_thetagt20 = np.array([4.0754238481177705, 4.193693485630508,  5.14209420056253,   6.114996249971468,  7.7376807326481645,
                               8.551663213602291,  11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604,
                               25.04088272634407,  28.853696411302344, 34.72538105333071,  40.21218694947545,  46.07344239520299,
                               50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047,  74.82378572939959,
                               89.4528387422834,   93.18228303096758,  92.51751129204555,  98.80228884380018,  111.17537347472128,
                               120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185,
                               177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696])

# Number of containment regions
nregions = 5

# Array holding endpoints for each longitudinal segment
segment_ends = np.array([[0, 5], [6, 16], [17, 33]])
nsegments = segment_ends.shape[0]

# Class for storing hit data
class HitData:

    def __init__(self, position = None, layer = None):

        self.position = position
        self.layer = layer


###################################
# Miscellaneous functions
###################################

# Function to get the position of a hit
def get_position(hit):

    # Two sets of method names for different hit types
    try: return np.array([hit.getPosition()[0], hit.getPosition()[1], hit.getPosition()[2]])
    except AttributeError: return np.array([hit.getXPos(), hit.getYPos(), hit.getZPos()])

# Function to get the layer z of a ECal hit
def get_layer_z(hit):

    return ecal_layerZs[get_ecal_layer(hit)]

# Function to get the momentum of a hit
def get_momentum(hit):

    return np.array([hit.getMomentum()[0], hit.getMomentum()[1], hit.getMomentum()[2]])

# Function to make a linear projection
def lin_proj(x, u, z):

    x1, y1, z1 = x
    ux, uy, uz = u

    if uz == 0: return np.full(3, np.nan)

    tc = (z - z1)/uz
    if tc < 0: return np.full(3, np.nan)

    x2 = ux*tc + x1
    y2 = uy*tc + y1
    z2 = z

    return np.array([x2, y2, z2])

# Function to get projected xy-intercepts
def get_intercepts(x, u, zs):

    return np.array([lin_proj(x, u, z)[0:2] for z in zs])

# Function to normalize a vector
def normalize(u):

    norm = np.linalg.norm(u)
    if norm == 0: return u

    return u/norm

# Function to calculate the distance between two points
def distance(x, y):

    return np.sqrt(np.sum((x - y)**2))

# Function to calculate the distance between a point and a line
def point_line_dist(x, y1, y2):

    norm = np.linalg.norm(y1 - y2)
    if norm == 0: return np.sqrt(np.sum((x - y1)**2))

    return np.linalg.norm(np.cross(x - y1, y1 - y2))/norm

# Function to calculate the distance between two lines
def line_line_dist(x1, x2, y1, y2):

    cross = np.cross(x1 - x2, y1 - y2)
    norm = np.linalg.norm(cross)

    if norm == 0:

        xnorm = np.linalg.norm(x1 - x2)
        ynorm = np.linalg.norm(y1 - y2)

        if xnorm == 0 and ynorm == 0: return np.sqrt(np.sum((x1 - y1)**2))
        elif ynorm == 0: return np.linalg.norm(np.cross(x1 - x2, x1 - y1))/xnorm

        return np.linalg.norm(np.cross(x1 - y1, y1 - y2))/ynorm

    return abs(np.dot(cross, x1 - y1)/norm)

# Function to calculate the angle between two vectors
def angle(u, v, units = 'radians'):

    uhat = normalize(u)
    vhat = normalize(v)

    if units == 'radians': return np.arccos(np.dot(uhat, vhat))
    elif units == 'degrees': return (180/np.pi)*np.arccos(np.dot(uhat, vhat))


##############################
# Hit ID information
##############################

# Function to get the layer ID of a ECal hit
def get_ecal_layer(hit):

    return (hit.getID() >> ecal_LAYER_SHIFT) & ecal_LAYER_MASK

# Function to get the module ID of a ECal hit
def get_ecal_module(hit):

    return (hit.getID() >> ecal_MODULE_SHIFT) & ecal_MODULE_MASK

# Function to get the cell ID of a ECal hit
def get_ecal_cell(hit):

    return (hit.getID() >> ecal_CELL_SHIFT) & ecal_CELL_MASK

# Function to get the section ID of a HCal hit
def get_hcal_section(hit):

    return (hit.getID() >> hcal_SECTION_SHIFT) & hcal_SECTION_MASK

# Function to get the layer ID of a HCal hit
def get_hcal_layer(hit):

    return (hit.getID() >> hcal_LAYER_SHIFT) & hcal_LAYER_MASK

# Function to get the strip ID of a HCal hit
def get_hcal_strip(hit):

    return (hit.getID() >> hcal_STRIP_SHIFT) & hcal_STRIP_MASK


#########################################
# Scoring plane hit information
#########################################

# Function to get the electron at the target scoring plane
def get_ele_targ_sp_hit(target_sp_hits):

    pmax = 0
    target_sp_hit = None
    for hit in target_sp_hits:

        position = get_position(hit)
        momentum = get_momentum(hit)

        if abs(position[2] - sp_trigger_pad_down_l2_z) > 0.5*sp_thickness or\
               momentum[2] <= 0 or\
               hit.getTrackID() != 1 or\
               hit.getPdgID() != 11:
            continue

        p = np.linalg.norm(momentum)
        if p > pmax:
            target_sp_hit = hit
            pmax = p

    return target_sp_hit

# Function to get the electron at the ECal scoring plane
def get_ele_ecal_sp_hit(ecal_sp_hits):

    pmax = 0
    ecal_sp_hit = None
    for hit in ecal_sp_hits:

        position = get_position(hit)
        momentum = get_momentum(hit)

        if abs(position[2] - sp_ecal_front_z) > 0.5*sp_thickness or\
               momentum[2] <= 0 or\
               hit.getTrackID() != 1 or\
               hit.getPdgID() != 11:
            continue

        p = np.linalg.norm(momentum)
        if p > pmax:
            ecal_sp_hit = hit
            pmax = p

    return ecal_sp_hit

# Function to get the photon at the target scoring plane
def get_pho_targ_sp_hit(target_sp_hits):

    pmax = 0
    target_sp_hit = None
    for hit in target_sp_hits:

        position = get_position(hit)
        momentum = get_momentum(hit)

        if abs(position[2] - sp_trigger_pad_down_l2_z) > 0.5*sp_thickness or\
               momentum[2] <= 0 or\
               not (hit.getPdgID() in [-22, 22]):
            continue

        p = np.linalg.norm(momentum)
        if p > pmax:
            target_sp_hit = hit
            pmax = p

    return target_sp_hit

# Function to get the photon at the ECal scoring plane
def get_pho_ecal_sp_hit(ecal_sp_hits):

    pmax = 0
    ecal_sp_hit = None
    for hit in ecal_sp_hits:

        position = get_position(hit)
        momentum = get_momentum(hit)

        if abs(position[2] - sp_ecal_front_z) > 0.5*sp_thickness or\
               momentum[2] <= 0 or\
               not (hit.getPdgID() in [-22, 22]):
            continue

        p = np.linalg.norm(momentum)
        if p > pmax:
            ecal_sp_hit = hit
            pmax = p

    return ecal_sp_hit

# Function to infer the photon's information at the target scoring plane
def infer_pho_targ_sp_hit(target_sp_hit):

    position = get_position(target_sp_hit)
    momentum = get_momentum(target_sp_hit)

    return position, np.array([0., 0., 4000.]) - momentum
