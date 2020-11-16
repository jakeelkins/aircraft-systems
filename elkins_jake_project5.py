import numpy as np
import math
import pandas as pd
import sys
sys.path.append('./project5_data/project5_data/LAMBDA_Python_1.0')
import LAMBDA

# get ephem data given
# list of what to call columns:
headerlist = ['prn', 'M0', 'delta_n', 'e', 'sqrt_a', 'Loa', 'i', 'perigee',
              'ra_rate', 'i_rate', 'Cuc', 'Cus', 'Crc', 'Crs', 'Cic', 'Cis',
              'Toe', 'IODE', 'GPS_week', 'Toc', 'Af0', 'Af1', 'Af2', '0']
eph_data = pd.read_csv('./project5_data/project5_data/gps_ephem.txt',
                       header=None, names=headerlist)

# SV pos function. need to input a time and SV PRN, get the
# SVs position at that time in ECEF


def _getSatPos(satPRN, request_time):

    # get ephem data given
    # list of what to call columns:
    headerlist = ['prn', 'M0', 'delta_n', 'e', 'sqrt_a', 'Loa', 'i',
                  'perigee', 'ra_rate', 'i_rate', 'Cuc', 'Cus', 'Crc', 'Crs',
                  'Cic', 'Cis', 'Toe', 'IODE', 'GPS_week', 'Toc', 'Af0',
                  'Af1', 'Af2', '0']
    eph_data = pd.read_csv('./project5_data/project5_data/gps_ephem.txt',
                           header=None, names=headerlist)

    eph_data['prn'] = eph_data['prn'].round(decimals=0)

    # find closest time to given request time for this prn
    idx_to_use = (abs(eph_data[eph_data['prn'] == satPRN]
                      ['Toe'] - request_time)).idxmin()

    # ---GPS constants---
    # mu of earth
    mu_e = 3.986005e+14

    # speed of light
    c = 2.99792458e+8

    # earth rotation rate
    OmegaDot_e = 7.2921151467e-5

    # pi
    pi = 3.1415926535898

    # -------------------
    # -------------get params initial------------------

    t0 = 0

    # reference time of ephemeris (s)
    toe = eph_data.loc[0, 'Toe']

    # root SMA. note--changes slightly over time. take first val
    sqrtA = eph_data.loc[t0, 'sqrt_a']

    # NaN catcher
    while math.isnan(sqrtA):
        for i in range(len(eph_data)):
            sqrtA = eph_data.loc[i, 'sqrt_a']

    # eccentricity. note--changes slightly over time. take first val
    ecc = eph_data.loc[t0, 'e']

    # NaN catcher
    while math.isnan(ecc):
        for i in range(len(eph_data)):
            ecc = eph_data.loc[i, 'e']

    # inclination at toe
    i0 = eph_data.loc[t0, 'i']

    # NaN catcher
    while math.isnan(i0):
        for i in range(len(eph_data)):
            i0 = eph_data.loc[i, 'i']

    # long. of ascending node at toe
    Omega0 = eph_data.loc[t0, 'Loa']

    # NaN catcher
    while math.isnan(Omega0):
        for i in range(len(eph_data)):
            Omega0 = eph_data.loc[i, 'Loa']

    # arg. of perigee at toe
    omega0 = eph_data.loc[t0, 'perigee']

    # NaN catcher
    while math.isnan(omega0):
        for i in range(len(eph_data)):
            omega0 = eph_data.loc[i, 'perigee']

    # mean anomaly at toe
    M0 = eph_data.loc[t0, 'M0']

    # NaN catcher
    while math.isnan(M0):
        for i in range(len(eph_data)):
            M0 = eph_data.loc[t0, 'M0']

    # rate of change of inclination
    IDOT = eph_data.loc[t0, 'i_rate']

    # NaN catcher
    while math.isnan(IDOT):
        for i in range(len(eph_data)):
            IDOT = eph_data.loc[i, 'i_rate']

    # rate of change of LAN
    OmegaDot = eph_data.loc[t0, 'ra_rate']

    # NaN catcher
    while math.isnan(OmegaDot):
        for i in range(len(eph_data)):
            OmegaDot = eph_data.loc[i, 'ra_rate']

    # mean motion correction
    DeltaN = eph_data.loc[t0, 'delta_n']

    # NaN catcher
    while math.isnan(DeltaN):
        for i in range(len(eph_data)):
            DeltaN = eph_data.loc[i, 'delta_n']

    # corrections
    # cosine correction to arg of latitude
    Cuc = eph_data.loc[t0, 'Cuc']

    # NaN catcher
    while math.isnan(Cuc):
        for i in range(len(eph_data)):
            Cuc = eph_data.loc[i, 'Cuc']

    # sine correction to arg of latitude
    Cus = eph_data.loc[t0, 'Cus']

    # NaN catcher
    while math.isnan(Cus):
        for i in range(len(eph_data)):
            Cus = eph_data.loc[i, 'Cus']

    # cosine correction to orbit radius
    Crc = eph_data.loc[t0, 'Crc']

    # NaN catcher
    while math.isnan(Crc):
        for i in range(len(eph_data)):
            Crc = eph_data.loc[i, 'Crc']

    # sine correction to orbit radius
    Crs = eph_data.loc[t0, 'Crs']

    # NaN catcher
    while math.isnan(Crs):
        for i in range(len(eph_data)):
            Crs = eph_data.loc[i, 'Crs']

    # cosine correction to inclination
    Cic = eph_data.loc[t0, 'Cic']

    # NaN catcher
    while math.isnan(Cic):
        for i in range(len(eph_data)):
            Cic = eph_data.loc[i, 'Cic']

    # sine correction to inclination
    Cis = eph_data.loc[t0, 'Cis']

    # NaN catcher
    while math.isnan(Cis):
        for i in range(len(eph_data)):
            Cis = eph_data.loc[i, 'Cis']

    # -------------at timestep k:------------------

    # reference time of ephemeris (s)
    t = eph_data.loc[idx_to_use, 'Toe']

    # root SMA. note--changes slightly over time, gotta check for NaN
    sqrtAx = eph_data.loc[idx_to_use, 'sqrt_a']

    if not math.isnan(sqrtAx):
        sqrtA = sqrtAx

    # eccentricity
    eccx = eph_data.loc[idx_to_use, 'e']

    if not math.isnan(eccx):
        ecc = eccx

    # rate of change of inclination
    IDOTx = eph_data.loc[idx_to_use, 'i_rate']

    if not math.isnan(IDOTx):
        IDOT = IDOTx

    # rate of change of LAN
    OmegaDotx = eph_data.loc[idx_to_use, 'ra_rate']

    if not math.isnan(OmegaDotx):
        OmegaDot = OmegaDotx

    # mean motion correction
    DeltaNx = eph_data.loc[idx_to_use, 'delta_n']

    if not math.isnan(DeltaNx):
        DeltaN = DeltaNx

    # corrections
    # cosine correction to arg of latitude
    Cucx = eph_data.loc[idx_to_use, 'Cuc']

    if not math.isnan(Cucx):
        Cuc = Cucx

    # sine correction to arg of latitude
    Cusx = eph_data.loc[idx_to_use, 'Cus']

    if not math.isnan(Cusx):
        Cus = Cusx

    # cosine correction to orbit radius
    Crcx = eph_data.loc[idx_to_use, 'Crc']

    if not math.isnan(Crcx):
        Crc = Crcx

    # sine correction to orbit radius
    Crsx = eph_data.loc[idx_to_use, 'Crs']

    if not math.isnan(Crsx):
        Crs = Crsx

    # cosine correction to inclination
    Cicx = eph_data.loc[idx_to_use, 'Cic']

    if not math.isnan(Cicx):
        Cic = Cicx

    # sine correction to inclination
    Cisx = eph_data.loc[idx_to_use, 'Cis']

    if not math.isnan(Cisx):
        Cis = Cisx

    # ------------------alg start------------------------

    # get SMA
    a = sqrtA**2

    # corrected mean mot in rad/s
    n = np.sqrt(mu_e/(a**3)) + DeltaN

    # time since ref TOE
    t_k = t - toe

    # mean anomaly
    M_k = M0 + (n*t_k)

    # ---iteratively solve for eccentric anomaly:---

    E_k = M_k
    ratio = 1

    while abs(ratio) > 1e-8:

        E_error = E_k - ecc*np.sin(E_k) - M_k
        E_deriv = 1 - ecc*np.cos(E_k)

        ratio = E_error/E_deriv

        E_k = E_k - ratio

    E_k = E_k + ratio

    # get true anomaly with atan2 for quadrant issues
    sin_nu_k = (np.sqrt(1 - ecc**2)*np.sin(E_k))/(1 - ecc*np.cos(E_k))
    cos_nu_k = (np.cos(E_k) - ecc)/(1 - ecc*np.cos(E_k))

    # true anomaly
    nu_k = np.arctan2(sin_nu_k, cos_nu_k)

    # nominal arg of latitude
    phi_k = nu_k + omega0

    # ----corrections----
    # correct arg of latitude
    u_k = phi_k + Cus*np.sin(2*phi_k) + Cuc*np.cos(2*phi_k)

    # get corrected radius
    r_k = a*(1 - ecc*np.cos(E_k)) + Crs*np.sin(2*phi_k) + Crc*np.cos(2*phi_k)

    # corrected inclination
    i_k = i0 + IDOT*t_k + Cis*np.sin(2*phi_k) + Cic*np.cos(2*phi_k)
    # -------

    # corrected long. of AN (set ref time as zero)
    Omega_k = Omega0 + (OmegaDot - OmegaDot_e)*t_k  # - (OmegaDot_e*toe)

    # ----position----
    # orbital in-plane pos in ECI coordinates
    x_prime_k = r_k*np.cos(u_k)
    y_prime_k = r_k*np.sin(u_k)

    # ECEF WGS84 x-y-z coords (position of satellite at this time k):
    x_k = x_prime_k*np.cos(Omega_k) - y_prime_k*np.cos(i_k)*np.sin(Omega_k)
    y_k = x_prime_k*np.sin(Omega_k) + y_prime_k*np.cos(i_k)*np.cos(Omega_k)
    z_k = y_prime_k*np.sin(i_k)

    return (x_k, y_k, z_k)

# BOOM!!!!


# get rover data given
rover = pd.read_csv('./project5_data/project5_data/rover.txt', header=None)

# get base data given
base = pd.read_csv('./project5_data/project5_data/base.txt', header=None)

# --constants--
# speed of light
c = 299792458   # m
# L1 freq
L1freq = 1575.42e+6   # Hz
# variance of carrier phase
sigma_phi = 0.0025   # cycles

wavelength = c/L1freq

# for fixing ambig later
baseline_true = 0.36   # m

# column indices for later
rec_prn = 0
rec_snr = 1
rec_csc = 2
rec_pr = 3
rec_cp = 4
rec_skip = 5

# number of columns not sat data specific
rec_data = 12  # python adaptation
# %%
# ----align times of base and rover----
# here we just delete the misaligned times
# record the new consistent times in this df
# base
save_df_base = base.copy(deep=True)
save_idx_base = []

# rover
save_df_rover = rover.copy(deep=True)
save_idx_rover = []

for i in range(len(base)):
    for j in range(len(rover)):
        # check if the TOW and weeknum both matchup
        if (base.loc[i, 0] == rover.loc[j, 0]) and\
           (base.loc[i, 1] == rover.loc[j, 1]):
            # use all these indices to drop
            if i not in save_idx_base:
                save_idx_base.append(i)
            if j not in save_idx_rover:
                save_idx_rover.append(j)

idx_base = base.index.tolist()
idx_rover = rover.index.tolist()

# checks which indices we drop (which arent in the savelist)
drop_idx_base = np.setdiff1d(idx_base, save_idx_base)
drop_idx_rover = np.setdiff1d(idx_rover, save_idx_rover)

# reassign with dropped vals
base = base.drop(drop_idx_base, axis='index', inplace=False)\
        .reset_index(drop=True)
rover = rover.drop(drop_idx_rover, axis='index', inplace=False)\
        .reset_index(drop=True)

print(f'entries dropped from:\n base:\
      {len(drop_idx_base)}\n rover: {len(drop_idx_rover)}')

# typecast all the measurement columns bc they read in as strings
for col in range(rec_data, rover.shape[1]):
    rover.loc[:, col] = pd.to_numeric(rover.loc[:, col], errors='coerce')

for col in range(rec_data, base.shape[1]):
    base.loc[:, col] = pd.to_numeric(base.loc[:, col], errors='coerce')

# ---use given GPS ToW in seconds to get static and motion portions---
static_tow_start = 417136.
static_tow_end = 417398.

kinematic_tow_end = 417885.

# get the indices of starts and stops of both
static_start_idx = base[base[0] == static_tow_start].index.to_list()[0]
static_end_idx = base[base[0] == static_tow_end].index.to_list()[0]
kinematic_end_idx = base[base[0] == kinematic_tow_end].index.to_list()[0]

# drop data we dont need (from start to static timeframe)
base = base.iloc[static_start_idx:].reset_index(drop=True)
rover = rover.iloc[static_start_idx:].reset_index(drop=True)

# ---find the sats we want to drop out---
# compare cycle slip counter in data for each sat
sats_used = 0

# sat_array = pd.DataFrame({})
sat_array = []

# idea is if the cycle slip counter is the same in beginning
# as end there were no slips in the timeframe
for satnum in range(32):
    if (rover.loc[0, rec_data + satnum*rec_skip + rec_csc] ==
        rover.loc[rover.index[-1], rec_data + satnum*rec_skip + rec_csc])\
        and (base.loc[0, rec_data + satnum*rec_skip + rec_csc] ==
             base.loc[rover.index[-1], rec_data + satnum*rec_skip + rec_csc]):

        sats_used += 1

        sat_array.append(satnum)

print(f'sats used for calc: {sats_used}')
print(f'sat PRNs: {sat_array}')
# %%
# for each sat, get the columns we need and concat to a new dataframe
base_pseudorange = pd.DataFrame({})
base_carrier_phase = pd.DataFrame({})
rover_pseudorange = pd.DataFrame({})
rover_carrier_phase = pd.DataFrame({})

for sat_prn_num in sat_array:
    base_pseudorange = pd.concat([base_pseudorange,
                                  base.loc[:, rec_data + sat_prn_num*rec_skip
                                           + rec_pr]], sort=False, axis=1)
    rover_pseudorange = pd.concat([rover_pseudorange,
                                   rover.loc[:, rec_data
                                             + sat_prn_num*rec_skip
                                             + rec_pr]], sort=False, axis=1)

    base_carrier_phase = pd.concat([base_carrier_phase, base.loc[:,
                                    rec_data +
                                    sat_prn_num*rec_skip +
                                    rec_pr]], sort=False, axis=1)
    rover_carrier_phase = pd.concat([rover_carrier_phase,
                                     rover.loc[:, rec_data +
                                               sat_prn_num*rec_skip +
                                               rec_pr]], sort=False, axis=1)

    # do the indices need reset? columns do.

base_pseudorange.columns = [i for i in range(base_pseudorange.shape[1])]
base_carrier_phase.columns = [i for i in range(base_carrier_phase.shape[1])]
rover_pseudorange.columns = [i for i in range(rover_pseudorange.shape[1])]
rover_carrier_phase.columns = [i for i in range(rover_carrier_phase.shape[1])]

# compute LOS vectors
# need big time list
timelist = rover.loc[:, 0].to_list()

los_data = pd.DataFrame({}, columns=[i for i in range(len(sat_array))])

# for every timestep
for i in range(len(rover)):

    request_time = timelist[i]
    los_vecs = []

    for satPRN in sat_array:

        curr_sat_pos = np.array(_getSatPos(satPRN+1, request_time))
        curr_rec_pos = np.array([base.iloc[i, 2], base.iloc[i, 3],
                                 base.iloc[i, 4]])

        # get unit vector of LOS from base to sat
        curr_los = (curr_rec_pos - curr_sat_pos) /\
            (np.linalg.norm(curr_rec_pos - curr_sat_pos))

        los_vecs.append(tuple(curr_los))

    los_data = los_data.append([los_vecs])
    los_data = los_data.reset_index(drop=True)

los_data.columns = sat_array
los_data['time'] = timelist
# %%
# ---double-differenced carrier phase measurements---

# get mean base pos in ECEF for reference for conversion
# only use the static part:
static_indices = base.loc[:static_end_idx, :].index

ecef_ref_x = np.mean(base.loc[static_indices, 2])
ecef_ref_y = np.mean(base.loc[static_indices, 3])
ecef_ref_z = np.mean(base.loc[static_indices, 4])

# and vectorize
ecef_ref = [ecef_ref_x, ecef_ref_y, ecef_ref_z]

# we now need max elevations for timesteps (use project 3 code here)
# flattening
f = 1/298.257223563

# eccentricity
e = np.sqrt(f*(2 - f))

# equatorial radius (m)
R_0 = 6378137

# polar radius (m)
R_p = R_0*(1 - f)

# ---transformations for base pos ---
x = ecef_ref[0]
y = ecef_ref[1]
z = ecef_ref[2]

# longitude:
lambdax = np.arctan2(y, x)

# ---getting altitude and latitude---
# intermediate terms
p = np.sqrt(x**2 + y**2)

E = np.sqrt(R_0**2 - R_p**2)

F = 54*(R_p*z)**2

G = p**2 + ((1 - e**2)*z**2) - (e*E)**2

c = ((e**4)*F*p**2)/(G**3)

s = (1 + c + np.sqrt(c**2 + 2*c))**(1/3)

P = (F/(3*G**2))/((s + (1/s) + 1)**2)

Q = np.sqrt(1 + 2*P*e**4)

k1 = -(P*p*e**2)/(1 + Q)

k2 = (0.5*R_0**2)*(1 + (1/Q))

k3 = -P*(1 - e**2)*((z**2)/(Q*(1 + Q)))

k4 = -0.5*P*p**2

k5 = p - (e**2)*(k1 + np.sqrt(k2 + k3 + k4))

U = np.sqrt(k5**2 + z**2)

V = np.sqrt(k5**2 + (1 - e**2)*z**2)

# finally, altitude:
h = U*(1 - ((R_p**2)/(R_0*V)))

# two more intermediates
z_0 = (z*R_p**2)/(R_0*V)

e_p = R_0*e/R_p

# latitude:
phi = np.arctan((z + z_0*e_p**2)/p)

# LLA for base pos avg:
base_lla = (phi, lambdax, h)
# -------------

# now go to NED, with Base pos as reference pos
R = np.array([
    [-np.sin(phi)*np.cos(lambdax), -np.sin(lambdax),
     -np.cos(phi)*np.cos(lambdax)],
    [-np.sin(phi)*np.sin(lambdax), np.cos(lambdax),
     -np.cos(phi)*np.sin(lambdax)],
    [np.cos(phi), 0, -np.sin(phi)]
])


# ----calculate line-of-sight (LOS) vector for each space
# vehicle from Base receiver
# utilize LLA of Base pos. will need to normalize

LOS_data_ned = pd.DataFrame({})

# loop over SV ECEF pos DF and do the vector math, then record in new DF
for sv in sat_array:
    los_data_ned = []
    for i in range(len(los_data)):
        # change loop to get los ecef from los_data

        los_ecef = np.array(los_data.loc[i, sv])

        los_ned = np.matmul(R.transpose(), los_ecef.transpose())

        los_x = los_ned[0]
        los_y = los_ned[1]
        los_z = los_ned[2]

        '''#get magnitude
        los_mag = np.sqrt(los_x**2 + los_y**2 + los_z**2)

        # normalize
        nlos_x = los_x/los_mag
        nlos_y = los_y/los_mag
        nlos_z = los_z/los_mag'''

        # record
        los_i = (los_x, los_y, los_z)
        los_data_ned.append(los_i)
    LOS_data_ned[sv] = los_data_ned


# now get elevation angle for each sv
# make vec for north-east plane, dot the vec in NED to get elev.
sv_elev_data = pd.DataFrame({})

for sv in sat_array:
    elev_data = []
    for i in range(len(LOS_data_ned)):
        currvec = LOS_data_ned.loc[i, sv]

        N = currvec[0]
        E = currvec[1]
        D = currvec[2]

        NED_vec = np.array([N, E, D])

        NE_plane_vec = np.array([N, E, 0.])

        cos_elev = np.dot(NED_vec, NE_plane_vec) /\
            (np.linalg.norm(NED_vec) * np.linalg.norm(NE_plane_vec))

        elev_angle = np.arccos(cos_elev)

        elev_data.append(elev_angle)
    sv_elev_data[sv] = elev_data


# find 'optimal' sat by max average elev angle over static period
max_avg_elev_angle = max(np.mean(sv_elev_data.loc[:static_end_idx, :]))
max_avg_elev_prn = np.mean(sv_elev_data.loc[:static_end_idx, :]).idxmax()

# ----single difference measurements----
sd_pseudorange = rover_pseudorange - base_pseudorange
sd_carrier_phase = rover_carrier_phase - base_carrier_phase

sd_pseudorange.columns = sat_array
sd_carrier_phase.columns = sat_array

# ---now do double difference based on our ref sat---
# these are for subtracting the dfs
dd_pseudo_sub_mat = pd.DataFrame({})
dd_cp_sub_mat = pd.DataFrame({})

for sv in sat_array:
    dd_pseudo_sub_mat[sv] = sd_pseudorange.loc[:, max_avg_elev_prn]
    dd_cp_sub_mat[sv] = sd_carrier_phase.loc[:, max_avg_elev_prn]

# and do the differencing
dd_pseudorange = sd_pseudorange - dd_pseudo_sub_mat
dd_carrier_phase = sd_carrier_phase - dd_cp_sub_mat

# %%
# ----DGPS solution----

dgps_soln_ned = pd.DataFrame({})

dgpsxlist = []
dgpsylist = []
dgpszlist = []

for i in range(len(rover)):
    HH = np.array([])
    yy = np.array([])

    for sv in sat_array:
        if sv != max_avg_elev_prn:
            # measurement matrix
            H = np.array(los_data.loc[i, sv]) - np.array(
                    los_data.loc[i, max_avg_elev_prn])

            # measurement vector
            y = dd_pseudorange.loc[i, sv]

            # append the measurements for giant measurement matrix
            if HH.size == 0:
                HH = H
                yy = y
            else:
                HH = np.block([
                    [HH],
                    [H]
                ])

                yy = np.block([
                    [yy],
                    [y]
                ])

    # now do the least squares solution in batch
    dgps_ecef = np.linalg.inv(HH.transpose()@HH)@HH.transpose()@yy

    # this is first row. needs appended and changed to NED
    # --- NED transfer---
    # flattening
    f = 1/298.257223563

    # eccentricity
    e = np.sqrt(f*(2 - f))

    # equatorial radius (m)
    R_0 = 6378137

    # polar radius (m)
    R_p = R_0*(1 - f)

    # ---transformations for base pos ---
    x = ecef_ref[0]
    y = ecef_ref[1]
    z = ecef_ref[2]

    # longitude:
    lambdax = np.arctan2(y, x)

    # ---getting altitude and latitude---
    # intermediate terms
    p = np.sqrt(x**2 + y**2)

    E = np.sqrt(R_0**2 - R_p**2)

    F = 54*(R_p*z)**2

    G = p**2 + ((1 - e**2)*z**2) - (e*E)**2

    c = ((e**4)*F*p**2)/(G**3)

    s = (1 + c + np.sqrt(c**2 + 2*c))**(1/3)

    P = (F/(3*G**2))/((s + (1/s) + 1)**2)

    Q = np.sqrt(1 + 2*P*e**4)

    k1 = -(P*p*e**2)/(1 + Q)

    k2 = (0.5*R_0**2)*(1 + (1/Q))

    k3 = -P*(1 - e**2)*((z**2)/(Q*(1 + Q)))

    k4 = -0.5*P*p**2

    k5 = p - (e**2)*(k1 + np.sqrt(k2 + k3 + k4))

    U = np.sqrt(k5**2 + z**2)

    V = np.sqrt(k5**2 + (1 - e**2)*z**2)

    # finally, altitude:
    h = U*(1 - ((R_p**2)/(R_0*V)))

    # two more intermediates
    z_0 = (z*R_p**2)/(R_0*V)

    e_p = R_0*e/R_p

    # latitude:
    phi = np.arctan((z + z_0*e_p**2)/p)

    # LLA for base pos avg:
    base_lla = (phi, lambdax, h)
    # -------------

    # now go to NED, with Base pos as reference pos
    R = np.array([
        [-np.sin(phi)*np.cos(lambdax), -np.sin(lambdax),
         -np.cos(phi)*np.cos(lambdax)],
        [-np.sin(phi)*np.sin(lambdax), np.cos(lambdax),
         -np.cos(phi)*np.sin(lambdax)],
        [np.cos(phi), 0, -np.sin(phi)]
    ])

    dgps_flat_ecef = np.array([dgps_ecef[0][0], dgps_ecef[1][0],
                               dgps_ecef[2][0]])
    dgps_ned = np.matmul(R.transpose(), dgps_flat_ecef.transpose())

    dgpsxlist.append(dgps_ned[0])
    dgpsylist.append(dgps_ned[1])
    dgpszlist.append(dgps_ned[2])

dgps_soln_ned['N'] = dgpsxlist
dgps_soln_ned['E'] = dgpsxlist
dgps_soln_ned['D'] = dgpsxlist
dgps_soln_ned['time'] = timelist

# ----integer ambiguity resolutions over static----
# %%
HHH = np.array([])
HHH_cand = np.array([])
yyy = np.array([])

for i in range(static_end_idx):
    HH = np.array([])
    yy = np.array([])

    for sv in sat_array:
        if sv != max_avg_elev_prn:
            # measurement matrix
            H = (1/wavelength)*(np.array(los_data.loc[i, sv]) - np.array(
                    los_data.loc[i, max_avg_elev_prn]))

            # measurement vector
            y = dd_carrier_phase.loc[i, sv]

            # append the measurements for giant measurement matrix
            if HH.size == 0:
                HH = H
                yy = y
            else:
                HH = np.block([
                    [HH],
                    [H]
                ])

                yy = np.block([
                    [yy],
                    [y]
                ])

    if HHH_cand.size == 0:
        HHH_cand = HH
        HHH = np.block([HH, -np.identity(len(sat_array)-1)])
        yyy = yy
    else:
        HHH_cand = np.block([
            [HHH_cand],
            [HH]
        ])

        HHH = np.block([
            [HHH],
            [HH, -np.identity(len(sat_array)-1)]
        ])

        yyy = np.block([
            [yyy],
            [yy]
        ])

# -------------------------------
# %%
# do least squares soln here
static_solution = np.linalg.inv(HHH.transpose()@HHH)@HHH.transpose()@yyy

static_x = static_solution[:3]

N_float = static_solution[3:]
# %%
# ---- Round integer ambiguity ----
N_round = N_float.round()
# %%
# ----geometry free----
gfree = (1/wavelength)*(dd_pseudorange) - dd_carrier_phase

# delete row of SV we use:
gfree = gfree.drop([max_avg_elev_prn], axis='columns')

N_gfree = gfree.mean(axis=0).round().to_numpy().reshape(sats_used-1, 1)
# %%
# ---LAMBDA---
# least squares ambiguity cecorr adjustment
single_R = 2*(sigma_phi**2)*(np.ones(
        (sats_used-1, sats_used-1))) + np.identity(sats_used-1)

R = np.zeros((int(len(yyy)/(sats_used-1)), int(len(yyy)/(sats_used-1))))

# for noise covar matrix by blocking them together
# NOTE: got singular matrix here
'''for i in range(sats_used):
    R[i:i+single_R.shape[0], i:i+single_R.shape[0]] = single_R'''

R = np.diag(2*(sigma_phi**2)*np.ones(yyy.shape[0]))

R_test = np.diag(2*(sigma_phi**2)*np.ones(N_float.shape[0]))

P = np.linalg.inv(HHH.transpose()@np.linalg.inv(R)@HHH)

P = np.delete(P, np.s_[:3], 0)
P = np.delete(P, np.s_[:3], 1)

afixed, sqnorm, Ps, Qzhat, Z, nfixed, mu = LAMBDA.main(N_float, R_test, 2, 3)

# least squares on each candidate
num_N = afixed.shape[0]
static_x_cand = np.zeros((3, num_N))
baseline_cand = np.zeros((1, num_N))

for i in range(num_N):
    static_x_cand[:, i] = ((np.linalg.inv(
            HHH_cand.transpose()@HHH_cand)@
        HHH_cand.transpose()).reshape(18, 311)@
        (yyy.reshape(int(yyy.shape[0]/6), 6) + np.matlib.repmat(afixed[:, 0],
         static_end_idx, 1))).mean(axis=0)

    baseline_cand[0, i] = np.linalg.norm(static_x_cand[:, i])

# EOF
