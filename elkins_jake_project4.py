import numpy as np
import pandas as pd
from scipy.linalg import expm
from pymap3d import geodetic2ned


def _skewSymm(vec):
    '''
    takes an input array and converts to cross-prod matrix
    '''
    out = np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])
    return out


g = 9.80665

corrector_counter = 0

init_euler = np.array([0, 0, 0])

gps_update_rate = 1.0

gps_ned_pos_stddev = 3.0
gps_ned_vel_stddev = 0.2

accel_bias_init = np.array([0.25, 0.077, -0.12])
accel_markov_bias_stddev = 0.0005*g
accel_bias_tau = 300
accel_output_noise_stddev = 0.12*g

gyro_bias_init = np.array([2.4e-4, -1.3e-4, 5.6e-4])
gyro_markov_bias_stddev = 0.3*np.pi/180
gyro_bias_tau = 300
gyro_output_noise_stddev = 0.95*np.pi/180

a = 6378137

f = 1/(298.257223563)
e = np.sqrt(f*(2 - f))

Omega_IE = 7.292115e-5
mu_E = 3.986004418e+14

imu_data = pd.read_csv('./project4_data/project4_data/imu.txt', sep=',',
                       header=None, names=['t', 'omega1', 'omega2', 'omega3',
                                           'alpha1', 'alpha2', 'alpha3'])

gps_pos_data = pd.read_csv('./project4_data/project4_data/gps_pos_lla.txt',
                           sep=',', header=None, names=['lat', 'long', 'alt'])

gps_vel_data = pd.read_csv('./project4_data/project4_data/gps_vel_ned.txt',
                           sep=',', header=None, names=['v_n', 'v_e', 'v_d'])

time_data = pd.read_csv('./project4_data/project4_data/time.txt', sep=',',
                        header=None, names=['t'])

# merge em together so we have a time col to help us out
gps_pos_data = pd.concat([time_data, gps_pos_data], axis=1)
gps_vel_data = pd.concat([time_data, gps_vel_data], axis=1)

# go ahead and switch lat and long to radians
gps_pos_data['lat'] = gps_pos_data['lat']*np.pi/180
gps_pos_data['long'] = gps_pos_data['long']*np.pi/180

init_orn = init_euler

# in LLA frame
init_lla = gps_pos_data.loc[0, ['lat', 'long', 'alt']].to_numpy()

# this one is in nav frame already.
init_velo = gps_vel_data.loc[0, ['v_n', 'v_e', 'v_d']].to_numpy()

# initialize error states for EKF:
# START LOOP
# --states--
orn = init_orn
ins_lla = init_lla
ins_v_ned = init_velo

accel_bias = accel_bias_init
gyro_bias = gyro_bias_init

# --error states--
pos_ned_error = gps_ned_pos_stddev*np.array([1.0, 1.0, 1.0])
v_ned_error = gps_ned_vel_stddev*np.array([1.0, 1.0, 1.0])
# assume 10x here like the code example i guess? need to test
orn_error = (1*np.pi/180)*np.array([1.0, 1.0, 1.0])
accel_bias_error = 1*(10*accel_markov_bias_stddev)*np.array([1.0, 1.0, 1.0])
gyro_bias_error = (1*gyro_markov_bias_stddev)*np.array([1.0, 1.0, 1.0])

# full error state vec
x_error = np.block([pos_ned_error, v_ned_error, orn_error,
                    accel_bias_error, gyro_bias_error])

# initial prediction matrix
P = 10*np.diag(np.square(x_error))

# loose H
H = np.block([np.identity(6), np.zeros((6, 9))])


ins_orn_list = []
ins_lla_list = []
ins_v_ned_list = []

accel_bias_list = []
gyro_bias_list = []

gps_lla_list = []
gps_v_ned_list = []

for t in range(1, len(time_data)):

    if t % 100 == 0:
        print(f'on iter num {t}')

    delta_t = round((time_data.loc[1, 't'] - time_data.loc[0, 't']), 2)

    ins_orn_list.append(orn)
    ins_lla_list.append(ins_lla)
    ins_v_ned_list.append(ins_v_ned)

    accel_bias_list.append(accel_bias)
    gyro_bias_list.append(gyro_bias)

    # ------------compute INS solution------------
    # I'll assume here we need the correction, so I'll do it in this section
    # TODO: compare results with correction and with assumption
    # transport rate = nonrot, flat Earth

    # ---attitude update---
    # first build DCM from body to nav, using current Euler angles
    phi = orn[0]
    theta = orn[1]
    psi = orn[2]

    # get ang velo of body in inertial frame from IMU measurements
    omega_BIB = imu_data.loc[t, ['omega1', 'omega2', 'omega3']].to_numpy()

    C_NB = np.array([
        [np.cos(theta)*np.cos(psi),
         np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
         np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
        [np.cos(theta)*np.sin(psi),
         np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),
         np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
        [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]
    ])

    # wsg84 ellipsoid calcs here
    R_N = (a*(1 - (e**2)))/((1 - (e**2)*((np.sin(ins_lla[0]))**2))**1.5)
    R_E = a/(np.sqrt((1 - (e**2)*((np.sin(ins_lla[0]))**2))))

    # AKA "transport rate", component by comp then form vector
    omega_NEN_1 = ins_v_ned[1]/(R_E + ins_lla[2])
    omega_NEN_2 = -ins_v_ned[0]/(R_N + ins_lla[2])
    omega_NEN_3 = -(ins_v_ned[1]*np.tan(ins_lla[0]))/(R_E + ins_lla[2])

    # vector. OOO 1e-6 in initial calcs. prob unnecessary
    omega_NEN = np.array([omega_NEN_1, omega_NEN_2, omega_NEN_3])

    # now need Earth rot. rate in rad/s
    omega_NIE = np.array([np.cos(ins_lla[0]), 0,
                          -np.sin(ins_lla[0])])*7.292115e-5

    # NED frame wrt inertial frame in just NED wrt ECEF plus ECEF wrt inertial
    omega_NIN = omega_NIE + omega_NEN

    # get omega correction
    omega_BIN = C_NB.transpose()@omega_NIN

    # ang velo of body to ned frame (needed)
    omega_BNB = omega_BIB - omega_BIN - gyro_bias

    # orientation rate of change
    A_psi = (1/np.cos(theta))*np.array([
        [1, np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta)],
        [0, np.cos(phi)*np.cos(theta), -np.sin(phi)*np.cos(theta)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    dot_orn = A_psi@(omega_BNB)

    # orientation/attitude Euler's integr. for next timestep
    orn = orn + (delta_t*dot_orn)

    # ---velo update---
    # get local gravity
    g0 = (9.7803253359 / (np.sqrt(1-(f*(2-f)*(np.sin(ins_lla[0]))**2)))) *\
        (1 + (0.0019311853 * (np.sin(ins_lla[0]))**2))

    # grav corrector
    c_h = 1 - 2*(1 + f + (((a**3)*(1 - f)*(Omega_IE**2))/mu_E))\
        * (ins_lla[2]/a) + 3*(ins_lla[2]/a)**2

    # vectorize
    g_N = np.array([0, 0, c_h*g0])

    # get current specific force in body frame from
    # IMU accelerometers (in m/s^2)
    f_B = imu_data.loc[t, ['alpha1', 'alpha2', 'alpha3']].to_numpy()\
        - accel_bias

    # convert to nav frame using C_NB
    f_N = C_NB@f_B

    # rate of change of velo (accel)
    dot_v_ned = (f_N - (_skewSymm((2*omega_NIE) - omega_NEN)@ins_v_ned)) + g_N

    # velo Euler's integr. for next timestep
    ins_v_ned = ins_v_ned + (delta_t*dot_v_ned)

    # ---lla position update---
    # rate of change of position
    T = np.array([
        [1/(R_N + ins_lla[2]), 0, 0],
        [0, 1/((R_E + ins_lla[2])*np.cos(ins_lla[0])), 0],
        [0, 0, -1]
    ])

    dot_lla = T@ins_v_ned

    # position Euler's integr. for next timestep
    ins_lla = ins_lla + (delta_t*dot_lla)

    # ---[!] IMU solution is here. {lla, v_ned, orn} [!]---

    # GPS solution for pos and velo. we don't use them in all cases, just have
    # it here for the list comparison for plotting and for if's to read
    gps_lla = gps_pos_data.loc[t, ['lat', 'long', 'alt']].to_numpy()
    gps_v_ned = gps_vel_data.loc[t, ['v_n', 'v_e', 'v_d']].to_numpy()

    gps_lla_list.append(gps_lla)
    gps_v_ned_list.append(gps_v_ned)

    # use vertical velo and position from previous
    # GPS update only, every ~1 second:
    if round((t-1) % round(1.0/delta_t)) == 0.0:
        gps_alt = gps_lla[2]
        gps_down_velo = gps_v_ned[2]

        ins_lla = gps_lla
        ins_v_ned = gps_v_ned

    # replace here with gps updates
    ins_lla[2] = gps_alt
    ins_v_ned[2] = gps_down_velo

    # propagate gyro bias and accel bias forward in time. constant.
    gyro_bias = gyro_bias
    accel_bias = accel_bias

    # ---- getting STM F and noise covariance K ----
    # now actually build the A state matrix. lots of blocks
    A = np.block([
        [-_skewSymm(omega_NEN), np.identity(3), np.zeros((3, 3)),
         np.zeros((3, 3)), np.zeros((3, 3))],
        [(np.linalg.norm(g_N)/a)*np.diag((-1, -1, 2)),
         -_skewSymm(2*omega_NIE + omega_NEN), _skewSymm(C_NB@f_B),
         C_NB, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)),
         -_skewSymm(omega_NIN), np.zeros((3, 3)), -C_NB],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)),
         -(1/accel_bias_tau)*np.identity(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)),
         np.zeros((3, 3)), -(1/gyro_bias_tau)*np.identity(3)]
    ])

    # and we need M, the noise gain matrix
    M = np.block([
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)),
         np.zeros((3, 3))],
        [C_NB, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), -C_NB, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3)]
    ])

    # now we form our discretized state matrix F using matrix exponential
    F = expm(A*delta_t)

    # need the power spectral densities (PSDs) of
    # Markov biases for accel and gyro
    # put them in loop for exact grav val
    accel_markov_bias_stddev = 0.0005*(np.linalg.norm(g_N))
    accel_output_noise_stddev = 0.12*(np.linalg.norm(g_N))

    # accel:
    psd_a = (2*(accel_markov_bias_stddev**2))/(accel_bias_tau)

    # gyro:
    psd_g = (2*(gyro_markov_bias_stddev**2))/(gyro_bias_tau)

    # form PSD matrix S
    S = np.block([
        [(accel_output_noise_stddev**2)*np.identity(3),
         np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), (gyro_output_noise_stddev**2)*np.identity(3),
         np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)),
         psd_a*np.identity(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)),
         psd_g*np.identity(3)]
    ])

    # ---get noise covar matrix Q---
    # use this as arg for E block martix
    E_exp_arg = np.block([
        [-A, M@S@(M.transpose())],
        [np.zeros(A.shape), A.transpose()]
    ])

    # now get E with discretization
    E = expm(E_exp_arg*delta_t)

    # block out E for Q calc
    E11 = E[:15, :15]
    E12 = E[:15, 15:]
    E21 = E[15:, :15]
    E22 = E[15:, 15:]

    # finally form Q
    Q = E22.transpose()@E12

    # update covariance
    P = A@P@A.transpose() + Q

    # ------correction step------
    # 1 Hz freq again:
    if round((t-1) % round(1.0/delta_t)) == 0.0:
        corrector_counter += 1
        # current LLA-->NED converter
        # use GPS here
        '''
        C_NLLA = np.array([
            [-np.sin(gps_lla[0])*np.cos(gps_lla[1]), -np.sin(gps_lla[1]),
            -np.cos(gps_lla[0])*np.cos(gps_lla[1])],
            [-np.sin(gps_lla[0])*np.sin(gps_lla[1]), np.cos(gps_lla[1]),
            -np.cos(gps_lla[0])*np.sin(gps_lla[1])],
            [np.cos(gps_lla[0]), 0, -np.sin(gps_lla[0])]
            ]).transpose()

        # get pos of GPS, INS in NED
        gps_pos_ned = C_NLLA@(gps_lla)
        ins_pos_ned = C_NLLA@(ins_lla)
        '''

        # ref pos is current INS pos estimate
        gps_pos_ned = np.array(geodetic2ned(gps_lla[0]*(180/np.pi),
                                            gps_lla[1]*(180/np.pi),
                                            gps_lla[2],
                                            ins_lla[0]*(180/np.pi),
                                            ins_lla[1]*(180/np.pi),
                                            ins_lla[2]))

        ins_pos_ned = np.array(geodetic2ned(ins_lla[0]*(180/np.pi),
                                            ins_lla[1]*(180/np.pi),
                                            ins_lla[2],
                                            ins_lla[0]*(180/np.pi),
                                            ins_lla[1]*(180/np.pi),
                                            ins_lla[2]))
        '''
        gps_pos_ned = np.array(geodetic2ned(gps_lla[0]*(180/np.pi),
            gps_lla[1]*(180/np.pi), gps_lla[2], init_lla[0]*(180/np.pi),
            init_lla[1]*(180/np.pi), init_lla[2]))
        ins_pos_ned = np.array(geodetic2ned(ins_lla[0]*(180/np.pi),
            ins_lla[1]*(180/np.pi), ins_lla[2], init_lla[0]*(180/np.pi),
            init_lla[1]*(180/np.pi), init_lla[2]))
        '''
        # position innovation
        pos_innov = ins_pos_ned - gps_pos_ned

        # velocity innovation
        velo_innov = ins_v_ned - gps_v_ned

        # vectorize
        state_innov = np.block([pos_innov, velo_innov])

        # form R with given stddevs of pos and velo
        Rdiag1 = gps_ned_pos_stddev*np.array([1.0, 1.0, 1.0])
        Rdiag2 = gps_ned_vel_stddev*np.array([1.0, 1.0, 1.0])

        R = np.diag(np.block([Rdiag1, Rdiag2]))

        # matrix inverse for gain equation
        gainInv = np.linalg.inv(H@P@H.transpose() + R)

        # ---Kalman gain---
        K = P@H.transpose()@gainInv

        # correct the error/innov
        error_corrected = K@state_innov

        # then correct the covariance
        P = (np.identity((K@H).shape[0]) - (K@H))@P

        # force covariance matrix symmetry? do we need this?
        P = 0.5*(P + P.transpose())

        # x_error = np.block([pos_ned_error,
        #   v_ned_error,
        #   orn_error, accel_bias_error, gyro_bias_error])
        pos_feedback = error_corrected[0:3]
        vel_feedback = error_corrected[3:6]
        orn_feedback = error_corrected[6:9]
        accel_feedback = error_corrected[9:12]
        gyro_feedback = error_corrected[12:15]

        # ---update INS states---
        # latitude
        ins_lla[0] = ins_lla[0] - (pos_feedback[0]/(R_N + ins_lla[2]))
        # longitude
        ins_lla[1] = ins_lla[1] - \
            (pos_feedback[1]/((R_E + ins_lla[2])*np.cos(ins_lla[0])))
        # altitude. gps only
        ins_lla[2] = gps_lla[2]  # + pos_feedback[2]

        # velocity
        ins_v_ned = ins_v_ned - vel_feedback
        # use gps for down velo
        ins_v_ned[2] = gps_v_ned[2]

        # accel bias
        accel_bias = accel_bias - accel_feedback
        # gyro bias
        gyro_bias = gyro_bias - gyro_feedback

        # attitude. need DCM again
        C_NB = (np.identity(3) + _skewSymm(orn_feedback))@C_NB

        # roll
        orn[0] = np.arctan(C_NB[2, 1]/C_NB[2, 2])
        # pitch
        orn[1] = -np.arcsin(C_NB[2, 0])
        # yaw
        # orn[2] = np.arctan(C_NB[1, 0]/C_NB[0, 0])
        orn[2] = np.arctan2(C_NB[1, 0], C_NB[0, 0])

# then plot here.
