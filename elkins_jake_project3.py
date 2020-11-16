import numpy as np
import os
import math
import pandas as pd

import georinex as gr

nav = gr.load('./project3_data/brdc2930.11n')

#---GPS constants---
# mu of earth
mu_e = 3.986005e+14

# speed of light
c = 2.99792458e+8

# earth rotation rate
OmegaDot_e = 7.2921151467e-5

# pi
pi = 3.1415926535898

#%%
#-------------------

# loop thru SVs, make dataset of coords for each SV
sv_pos_data = pd.DataFrame({})

for sv in nav.sv.data:
    #empty pos_data list for data saving
    pos_data = []

    # -------------get params initial------------------

    # initial time for indexing (in ns)
    t0 = nav.time[0].data.item(0)

    # reference time of ephemeris (s)
    toe = nav.time[0].data.item(0)/1e+9

    # root SMA. note--changes slightly over time. take first val
    sqrtA = nav.sel(sv=sv, time=t0).sqrtA.data.item(0)
    
    # NaN catcher
    while math.isnan(sqrtA):
        for i in range(len(nav.sel(sv=sv).sqrtA.data)):
            sqrtA = nav.sel(sv=sv).sqrtA.data.item(i)

    # eccentricity. note--changes slightly over time. take first val
    ecc = nav.sel(sv=sv, time=t0).Eccentricity.data.item(0)
    
    # NaN catcher
    while math.isnan(ecc):
        for i in range(len(nav.sel(sv=sv).Eccentricity.data)):
            ecc = nav.sel(sv=sv).Eccentricity.data.item(i)

    # inclination at toe
    i0 = nav.sel(sv=sv, time=t0).Io.data.item(0)
    
    # NaN catcher
    while math.isnan(i0):
        for i in range(len(nav.sel(sv=sv).Io.data)):
            i0 = nav.sel(sv=sv).Io.data.item(i)

    # long. of ascending node at toe
    Omega0 = nav.sel(sv=sv, time=t0).Omega0.data.item(0)
    
    # NaN catcher
    while math.isnan(Omega0):
        for i in range(len(nav.sel(sv=sv).Omega0.data)):
            Omega0 = nav.sel(sv=sv).Omega0.data.item(i)

    # arg. of perigee at toe
    omega0 = nav.sel(sv=sv, time=t0).omega.data.item(0)
    
    # NaN catcher
    while math.isnan(omega0):
        for i in range(len(nav.sel(sv=sv).omega.data)):
            omega0 = nav.sel(sv=sv).omega.data.item(i)

    # mean anomaly at toe
    M0 = nav.sel(sv=sv, time=t0).M0.data.item(0)
    
    # NaN catcher
    while math.isnan(M0):
        for i in range(len(nav.sel(sv=sv).M0.data)):
            M0 = nav.sel(sv=sv).M0.data.item(i)

    # rate of change of inclination
    IDOT = nav.sel(sv=sv, time=t0).IDOT.data.item(0)
    
    # NaN catcher
    while math.isnan(IDOT):
        for i in range(len(nav.sel(sv=sv).IDOT.data)):
            IDOT = nav.sel(sv=sv).IDOT.data.item(i)

    # rate of change of LAN
    OmegaDot = nav.sel(sv=sv, time=t0).OmegaDot.data.item(0)
    
    # NaN catcher
    while math.isnan(OmegaDot):
        for i in range(len(nav.sel(sv=sv).OmegaDot.data)):
            OmegaDot = nav.sel(sv=sv).OmegaDot.data.item(i)

    # mean motion correction
    DeltaN = nav.sel(sv=sv, time=t0).DeltaN.data.item(0)
    
    # NaN catcher
    while math.isnan(DeltaN):
        for i in range(len(nav.sel(sv=sv).DeltaN.data)):
            DeltaN = nav.sel(sv=sv).DeltaN.data.item(i)

    # corrections
    # cosine correction to arg of latitude
    Cuc = nav.sel(sv=sv, time=t0).Cuc.data.item(0)
    
    # NaN catcher
    while math.isnan(Cuc):
        for i in range(len(nav.sel(sv=sv).Cuc.data)):
            Cuc = nav.sel(sv=sv).Cuc.data.item(i)

    # sine correction to arg of latitude
    Cus = nav.sel(sv=sv, time=t0).Cus.data.item(0)
    
    # NaN catcher
    while math.isnan(Cus):
        for i in range(len(nav.sel(sv=sv).Cus.data)):
            Cus = nav.sel(sv=sv).Cus.data.item(i)

    # cosine correction to orbit radius
    Crc = nav.sel(sv=sv, time=t0).Crc.data.item(0)
    
    # NaN catcher
    while math.isnan(Crc):
        for i in range(len(nav.sel(sv=sv).Crc.data)):
            Crc = nav.sel(sv=sv).Crc.data.item(i)

    # sine correction to orbit radius
    Crs = nav.sel(sv=sv, time=t0).Crs.data.item(0)
    
    # NaN catcher
    while math.isnan(Crs):
        for i in range(len(nav.sel(sv=sv).Crs.data)):
            Crs = nav.sel(sv=sv).Crs.data.item(i)

    # cosine correction to inclination
    Cic = nav.sel(sv=sv, time=t0).Cic.data.item(0)
    
    # NaN catcher
    while math.isnan(Cic):
        for i in range(len(nav.sel(sv=sv).Cic.data)):
            Cic = nav.sel(sv=sv).Cic.data.item(i)

    # sine correction to inclination
    Cis = nav.sel(sv=sv, time=t0).Cis.data.item(0)
    
    # NaN catcher
    while math.isnan(Cis):
        for i in range(len(nav.sel(sv=sv).Cis.data)):
            Cis = nav.sel(sv=sv).Cis.data.item(i)

    # -------------at timestep k (loop here):------------------
    for k in range(len(nav.time.data)):

        # time for indexing (in ns)
        ti = nav.time[k].data.item(0)

        # reference time of ephemeris (s)
        t = nav.time[k].data.item(0)/1e+9

        # root SMA. note--changes slightly over time, gotta check for NaN
        sqrtAx = nav.sel(sv=sv, time=ti).sqrtA.data.item(0)

        if not math.isnan(sqrtAx):
            sqrtA = sqrtAx

        # eccentricity
        eccx = nav.sel(sv=sv, time=ti).Eccentricity.data.item(0)

        if not math.isnan(eccx):
            ecc = eccx

        # rate of change of inclination
        IDOTx = nav.sel(sv=sv, time=ti).IDOT.data.item(0)

        if not math.isnan(IDOTx):
            IDOT = IDOTx

        # rate of change of LAN
        OmegaDotx = nav.sel(sv=sv, time=ti).OmegaDot.data.item(0)

        if not math.isnan(OmegaDotx):
            OmegaDot = OmegaDotx

        # mean motion correction
        DeltaNx = nav.sel(sv=sv, time=ti).DeltaN.data.item(0)

        if not math.isnan(DeltaNx):
            DeltaN = DeltaNx

        # corrections
        # cosine correction to arg of latitude
        Cucx = nav.sel(sv=sv, time=ti).Cuc.data.item(0)

        if not math.isnan(Cucx):
            Cuc = Cucx

        # sine correction to arg of latitude
        Cusx = nav.sel(sv=sv, time=ti).Cus.data.item(0)

        if not math.isnan(Cusx):
            Cus = Cusx

        # cosine correction to orbit radius
        Crcx = nav.sel(sv=sv, time=ti).Crc.data.item(0)

        if not math.isnan(Crcx):
            Crc = Crcx

        # sine correction to orbit radius
        Crsx = nav.sel(sv=sv, time=ti).Crs.data.item(0)

        if not math.isnan(Crsx):
            Crs = Crsx

        # cosine correction to inclination
        Cicx = nav.sel(sv=sv, time=ti).Cic.data.item(0)

        if not math.isnan(Cicx):
            Cic = Cicx

        # sine correction to inclination
        Cisx = nav.sel(sv=sv, time=ti).Cis.data.item(0)

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
        Omega_k = Omega0 + (OmegaDot - OmegaDot_e)*t_k# - (OmegaDot_e*toe)

        # ----position----
        # orbital in-plane pos in ECI coordinates
        x_prime_k = r_k*np.cos(u_k)
        y_prime_k = r_k*np.sin(u_k)

        # ECEF WGS84 x-y-z coords (position of satellite at this time k):
        x_k = x_prime_k*np.cos(Omega_k) - y_prime_k*np.cos(i_k)*np.sin(Omega_k)
        y_k = x_prime_k*np.sin(Omega_k) + y_prime_k*np.cos(i_k)*np.cos(Omega_k)
        z_k = y_prime_k*np.sin(i_k)
        
        # save the info
        pos_data.append((x_k, y_k, z_k))
    
    sv_pos_data[sv] = pos_data

# -------get avg base position-------
base_ecef = pd.read_csv('./project3_data/data_base/ecef_rx0.txt', sep='\t', header=None)

base_ecef = base_ecef.drop(labels=[0], axis='index').drop_duplicates(0).reset_index(drop=True)

# calc like center of mass with timestep intervals. Use loop I guess

num_x = 0
num_y = 0
num_z = 0

denom = 0

for i in range(len(base_ecef)):
    # skip first val
    if i>0:
        delta_t = base_ecef.loc[i, 0] - base_ecef.loc[i-1, 0]
        
        x_i = base_ecef.loc[i, 2]
        y_i = base_ecef.loc[i, 3]
        z_i = base_ecef.loc[i, 4]
        
        num_x += x_i*delta_t
        num_y += y_i*delta_t
        num_z += z_i*delta_t
        
        denom += delta_t

# avg ECEF position of Base receiver
x_avg = num_x/denom
y_avg = num_y/denom
z_avg = num_z/denom

# -----------translate vectors----------

base_pos_avg = (x_avg, y_avg, z_avg)

# flattening
f = 1/298.257223563

# eccentricity
e = np.sqrt(f*(2 - f))

# equatorial radius (m)
R_0 = 6378137

# polar radius (m)
R_p = R_0*(1 - f)

# ---transformations for base pos ---
x = base_pos_avg[0]
y = base_pos_avg[1]
z = base_pos_avg[2]

# longitude:
lambdax = np.arctan2(y,x)

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
    [-np.sin(phi)*np.cos(lambdax), -np.sin(lambdax), -np.cos(phi)*np.cos(lambdax)],
    [-np.sin(phi)*np.sin(lambdax), np.cos(lambdax), -np.cos(phi)*np.sin(lambdax)],
    [np.cos(phi), 0, -np.sin(phi)]
])


# ----calculate line-of-sight (LOS) vector for each space vehicle from Base receiver
# utilize LLA of Base pos. will need to normalize

LOS_data_ned = pd.DataFrame({})

# loop over SV ECEF pos DF and do the vector math, and normalize (?), then record in new DF
for sv in nav.sv.data:
    los_data_ned = []
    for i in range(len(sv_pos_data)):
        
        # get vector to SV
        r_sv = sv_pos_data.loc[i, sv]
        
        x_sv = r_sv[0]
        y_sv = r_sv[1]
        z_sv = r_sv[2]
        
        # do the subtraction
        los_x_ecef = x_sv - x_avg
        los_y_ecef = y_sv - y_avg
        los_z_ecef = z_sv - z_avg
        
        r_ecef_ref = np.array([los_x_ecef, los_y_ecef, los_z_ecef])
        
        los_ned = np.matmul(R.transpose(), r_ecef_ref.transpose())
        
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

#%%

# now get elevation angle for each sv
# make vec for north-east plane, dot the vec in NED to get elev.
sv_elev_data = pd.DataFrame({})

for sv in nav.sv.data:
    elev_data = []
    for i in range(len(LOS_data_ned)):
        currvec = LOS_data_ned.loc[i, sv]

        N = currvec[0]
        E = currvec[1]
        D = currvec[2]

        NED_vec = np.array([N, E, D])

        NE_plane_vec = np.array([N, E, 0.])

        cos_elev = np.dot(NED_vec, NE_plane_vec)/(np.linalg.norm(NED_vec)*np.linalg.norm(NE_plane_vec))

        elev_angle = np.arccos(cos_elev)
        
        elev_data.append(elev_angle)
    sv_elev_data[sv] = elev_data

# getting max elev angle sat for each timestep
# just find max angle in each row of DF we formed (should be in [-90, 90])

# SV ID with max angle at each timestep list:
sv_max_elev_id_list = sv_elev_data.idxmax(axis=1).to_list()

sv_max_elev_angle_list = sv_elev_data.max(axis=1).to_list()

print(f'+----------------------------------------------+')
print(f'| ---- SV max elevation angle calculations --- |')
print(f'+----------------------------------------------+')
print(f'+--timestep--------SV ID---------angle (deg.)--+')
print(f'+----------------------------------------------+')
for i, sv in enumerate(sv_max_elev_id_list):
    print(f'| ---- {i} ---------- {sv} ------------ {round(sv_max_elev_angle_list[i]*180/pi, 1)}----- |')
print(f'+-----------------------------------------------+\n')



# now we select the ref satellites and read in their files and pseudoranges
# only have a set amount of sats in the file...which sats do we have?
# keep alg abstracted
filenames = os.listdir('./project3_data/data_rover')

sv_avail = []

for filename in filenames:
    result = re.findall(r'[a-z](\d+).', filename)
    
    sv_num = result[0]
    
    # pad with leading zero if single digit
    if len(sv_num) == 1:
        sv_num = '0'+sv_num
    
    # string of sats in the file
    sv_avail.append(sv_num)
    
# deep copy data for butchering
sv_elev_avail = sv_elev_data.copy(deep=True)

# make list so we can drop those columns in the new df
sv_not_avail = []

for sv_str in sv_elev_avail.columns.to_list():
    
    sv_num = sv_str[1:]

    if sv_num not in sv_avail:
        sv_not_avail.append('G'+sv_num)


# make new df so we can work with this
sv_elev_avail = sv_elev_avail.drop(labels=sv_not_avail, axis='columns')

# SV ID with max angle at each timestep list for availabale ones:
sv_max_elev_id_list2 = sv_elev_avail.idxmax(axis=1).to_list()

sv_max_elev_angle_list2 = sv_elev_avail.max(axis=1).to_list()

print(f'+----------------------------------------------+')
print(f'| --- SV max elev. angle avail. calculations -- |')
print(f'+----------------------------------------------+')
print(f'+--timestep--------SV ID---------angle (deg.)--+')
print(f'+----------------------------------------------+')
for i, sv in enumerate(sv_max_elev_id_list2):
    print(f'| ---- {i} ---------- {sv} ------------ {round(sv_max_elev_angle_list2[i]*180/pi, 1)}----- |')
print(f'+-----------------------------------------------+\n')

# make df for easy access:
sv_rover_ref = pd.DataFrame({'SV':sv_max_elev_id_list2, 'angle':sv_max_elev_angle_list2})

#%%










# **TODO: NEED switching timestep stuff here to indiv reference SVs






for interval in range(len(sv_rover_ref)):
    
    ref_sv = sv_rover_ref.loc[interval, 'SV']
    ref_sv_num = ref_sv[1:]
    if ref_sv_num[0] == '0':
        ref_sv_num = ref_sv_num[1:]

    # parse file we need for current ref SV
    icp_ref = pd.read_csv(f'./project3_data/data_rover/icp_sat{ref_sv_num}.txt', sep='\t', header=None)

    icp_ref = base_ecef.drop(labels=[0], axis='index').drop_duplicates(0).reset_index(drop=True)
    

# **TODO: read in pseudorange for single and double difference eq's

# **TODO: ILS

# **fit gaussian to errors, construct H to read off dilutions of precision