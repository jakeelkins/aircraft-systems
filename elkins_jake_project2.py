import numpy as np
from scipy import signal, linalg

A = np.array([
    [-2.382, 0.0, -30.1, 65.49, 0.0],
    [-0.702, -16.06, 0.872, 0.0, 0.0],
    [0.817, -16.65, -3.54, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    ])

B = np.array([
    [0.0, -7.41],
    [-36.3, -688.],
    [-0.673, -68.0],
    [0.0, 0.0],
    [0.0, 0.0],
    ])

C = np.identity(A.shape[0])

D = np.zeros(B.shape)

timestep = 0.01
outer_timestep = 1.0    # (s)

# -------wind--------
# model wind for -15, 0, 15 degrees
wind_angle = 0.0*np.pi/180
# resolve in Earth-frame
w_xe = 3.0*np.cos(wind_angle)
w_ye = 3.0*np.sin(wind_angle)

v_w = np.array([
    [w_xe],
    [w_ye]
])

# -----initial conditions-----
psi_initial = 70    # *np.pi/180
x_initial = -150
y_initial = 0
u_initial = 30    # (m/s)

y_hat_waypt = np.array([0, 1])

u_const = u_initial
v = 0

x = x_initial
y = y_initial
psi = psi_initial

psi_dot = 0.0

# -----------------------
gamma_h = psi
gamma_h_dot = psi_dot

# here, x is same as d_h
d_h = x

# psi settle time required
ts_psi = 10   # (s)

ts_h = 100   # (s). where does this get used??


dsystem = signal.cont2discrete((A, B, C, D), timestep, method='zoh')

dA = dsystem[0]
dB = dsystem[1]
dC = dsystem[2]
dD = dsystem[3]

# %% Section 2: inner attitude controller

# ---inner attitude controller development---

dA_inner = dA
dB_inner = dB
dC_inner = dC
dD_inner = dD

# pick Q and R based on control reqs
rho_inner = 1.5**-2
# based on allowed aileron/rudder deflections of 10 deg, Bryson's method

# Q_inner = np.diag([3**-2, 3**-2, 2**-2, 5**-2,
#                    5**-2, 1.0, 1.0, 1.0, 1.0, 1.0])
# Q_inner = np.diag([20**-2, 20**-2, 20**-2, 20**-2,
#                   20**-2, 20**-2, 20**-2, 20**-2, 20**-2, 20**-2])
# Q_inner = np.diag([3**-2, 3**-2, 2**-2, 10**-2, 10**-2,
#                  50**-2, 50**-2, 50**-2, 50**-2, 50**-2])
Q_inner = np.diag([12**-2, 12**-2, 12**-2, 14**-2, 11**-2, 50**-2, 50**-2,
                   50**-2, 50**-2, 50**-2])
R_inner = rho_inner*np.identity(dB_inner.shape[1])

# new augmented SS with integral error
A_aug = np.block([
             [dA_inner, np.zeros(dA_inner.shape)],
             [-timestep*np.identity(dA_inner.shape[0]),
              np.identity(dA_inner.shape[0])]
             ])

B_aug = np.block([
             [dB_inner],
             [np.zeros(dB_inner.shape)]
             ])

# assume cross weight matrix S = 0
P_inner = linalg.solve_discrete_are(A_aug, B_aug, Q_inner, R_inner)
K_inner = linalg.inv(B_aug.T@P_inner@B_aug + R_inner)@(B_aug.T@P_inner@A_aug)
# ----------------------------------------------

# %% Section 3: outer guidance system

# --------outer guidance pointing system development-------
# build SS model of line-tracker
tau_psi = ts_psi/3

# do we use tau+psi here or the one for d_h?
A_outer = np.array([
    [0., u_initial],
    [0., -1/tau_psi]
])

B_outer = np.array([
    [0.],
    [1/tau_psi]
])

C_outer = np.identity(A_outer.shape[0])

D_outer = np.zeros(B_outer.shape)

outer_dsystem = signal.cont2discrete((A_outer, B_outer, C_outer, D_outer),
                                     outer_timestep, method='zoh')

dA_outer = outer_dsystem[0]
dB_outer = outer_dsystem[1]
dC_outer = outer_dsystem[2]
dD_outer = outer_dsystem[3]

# need output of psi_des to be equal to 10 at the most.
# this was the sol'n based on a control of 10 deg

# rho_outer = (10*np.pi/180)**-2
rho_outer = (100)**-2
R_outer = rho_outer*np.identity(dB_outer.shape[1])

q_outer = (7)**-2
Q_outer = q_outer*np.identity(dA_outer.shape[0])

P_outer = linalg.solve_discrete_are(dA_outer, dB_outer, Q_outer, R_outer)

# phi_des = -dB_outer.T@P_outer@x_k_outer
# -------------------------------------------------------------
# %% Section 4: Simulation

# ----------------------simulation-----------------------

# init a few variables to be looped
e_prev = x_k = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
])

x_prev = x_k = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
])

x_k = np.array([
    [v],
    [0.0],
    [0.0],
    [0.0],
    [psi_initial],
])

x_k_outer = np.array([
    [d_h],
    [psi_initial],
])

# lists for recording plot vals
x_list = []
y_list = []
x_list.append(x)
y_list.append(y)

# -----------SIM LOOP---------
# think about: where do we record data for plotting
# is wind correct
# how long to simulate
# check settle times

print('starting sim loop...')
for i in range(1000):
    print(i)

    if i % 100 == 0:
        print(f'on loop {i}, inner control')

        # do outer loop calc and inner loop control
        # output phi desired
        psi_des = (-dB_outer.T@P_outer@x_k_outer).item()
        print(f'curr psi_des: {psi_des}')

        # propagate this outer SS model
        # do we propagate this or use the vals from other SS propagation?
        x_k_outer_dot = dA_outer@x_k_outer + dB_outer*psi_des

        # integrate (Eulers)
        x_k_outer = x_k_outer + outer_timestep*x_k_outer_dot

        # print(f'curr x_k_outer: {x_k_outer}')

    # ---------inner control---------
    # send it to the attitude controller
    x_des = np.array([[0.0], [0.0], [0.0], [0.0], [psi_des]])

    # block together matrices needed for output control
    e_ik = timestep*(x_des - x_k) + x_prev
    x_prev = x_k

    delta_x_k = x_k - x_des

    u_k = K_inner@np.block([[x_k], [e_ik]])
    # --------------------------------

    # propagate dynamics w/ current control
    x_k_dot = dA@x_k + dB@u_k

    # integrate (Euler's method)
    x_k = x_k + timestep*(x_k_dot)

    v = x_k[0].item()

    psi_rad = x_k[4].item()*np.pi/180

    dcm_k = np.array([
        [np.cos(psi_rad), -np.sin(psi_rad)],
        [np.sin(psi_rad), np.cos(psi_rad)]
        ])

    v_g = dcm_k@np.array([[u_const], [v]]) + v_w

    # position integration
    x = x + timestep*v_g[0].item()
    y = y + timestep*v_g[1].item()

    x_list.append(x)
    y_list.append(y)


print('sim done.')
