# -*- coding: utf-8 -*-
"""
AEM 517 Project 1 - Jake Elkins

This program takes a state-space model given and does the following,
organized by sections:
    1. Takes the given state-space model and discretizes for timestep of 0.1s
    2. Analyzes free-response stability
    3. Checks for controllability
    4. Designs a stabilizing state-feedback attitude controller with the
        following requirements:
            for a 1 deg step response for any angle:
            - settling time in less than 2 seconds
            - maximum overshoot less than 5%
    5. Simulates three 1 deg step responses (one for each (roll, pitch, yaw))
"""

import numpy as np
from scipy import signal, linalg
from sympy import Matrix
import plotly.graph_objects as go

# state space model
A = np.array([
    [-0.03, 0., 0.05, 0., 0., 0., 0., -9.8, 0.],
    [0., -0.11, 0., 0.065, 0., -119.1, 9.8, 0., 0.],
    [-0.18, 0., -1.4, 0., 120., 0., 0., 0., 0.],
    [0., -0.0074, 0., -1.85, 0., 0.25, 0., 0., 0.],
    [0.0021, 0., -0.27, 0., -6.14, 0., 0., 0., 0.],
    [0., 0.030, 0., -0.061, 0., -0.21, 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0.]
    ])
B = np.array([
    [0., 0., 0., 0.11],
    [0., 0., 3.69, 0.],
    [0., -12.8, 0., 0.0233],
    [2.31, 0., 0.26, 0.],
    [0., -6.48, 0., 0.],
    [-0.081, 0., -1.36, 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.]
    ])
C = np.identity(A.shape[0])
D = np.zeros(B.shape)

timestep = 0.1    # units of (s)


if __name__ == '__main__':
    # %% section 1: Takes the given state-space model and discretizes it for
    # a timestep of 0.1s

    # convert the system into SciPy's linear-time invariant (LTI) SS model
    system = signal.lti(A, B, C, D)

    # assume zero-order hold (input vector constant from t=0 to t=dt)
    dsystem = signal.cont2discrete((A, B, C, D), timestep, method='zoh')

    # %% section 2: Analyzes free-response stability

    # get eigenvals and eigen vectors of A
    A_eigenvals, A_V = linalg.eig(A)

    # get this matrix into sympy form to use jordan_form function
    A_matrix = Matrix(A)

    # convert A into Jordan canonical form
    A_Lambda = A_matrix.jordan_form(calc_transform=False)

    # find the magnitudes of the matrix parts
    # of Lambda of A (matrix of eigenvalues of A)
    # this is to analyze stability
    lambda_mag_matrix = np.absolute(A_Lambda)

    # quick loop to help with analyzing the eigenvalues in the JCF matrix
    over_1 = 0
    below_1 = 0
    equal_1 = 0

    for i in range(lambda_mag_matrix.shape[0]):

        curr_analysis = lambda_mag_matrix[i, i]

        if curr_analysis > 1:
            over_1 += 1
        elif curr_analysis < 1:
            below_1 += 1
        elif curr_analysis == 1:
            equal_1 += 1

    print(f'free response stability of A analyzed. found:\n\
            {over_1} e-vals > 1\n\
            {below_1} e-vals < 1\n\
            {equal_1} e-vals = 1.')

    if over_1 != 0:
        print('[!] this sytem is unstable, some e-vals magnitudes are > 1.\
              (for discrete time) [!]\n')

    # %% section 3: Checks for controllability

    if np.count_nonzero((A_Lambda - np.diag(np.diagonal(A_Lambda)))) == 0:
        # then A is diagonalizable
        print('A is diagnoalizable...building control matrix')
        # since A is diagonalizable, we can use this for LTI
        control_matrix = np.matmul(linalg.inv(A_V), B)
        print(f'control matrix: {control_matrix}')
    else:
        print('\n[!] A not diagonalizable. check process or entry [!]\n')

    # stabilizability doesnt need analyzed since system is controllable

    # %% section 4: 4. Designs a stabilizing state-feedback attitude controller

    # use poleplacement to get K feedback gain matrix
    # this one worked
    # poles_req = np.array([-0.01-0.1j, -0.01+0.1j, -0.01-0.1j, -0.01+0.1j,\
    # -0.01-0.1j, -0.01+0.1j, -0.01-0.1j, -0.01+0.1j, -0.01])
    # poles_req = np.array([-0.01, -0.01+0.1j, -0.01-0.1j, -0.01+0.1j,\
    # -0.01-0.1j, -0.01+0.1j, -0.01-0.1j, -0.01+0.1j, -0.01-0.1j])
    # not bad
    # poles_req = np.array([-0.01+0.1j, -0.01, -0.01-0.1j, -0.01+0.1j,\
    # -0.01-0.1j, -0.01+0.1j, -0.01-0.1j, -0.01+0.1j, -0.01-0.1j])
    # poles_req = np.array([-0.01+0.1j, 0.04, -0.01-0.1j, -0.01+0.1j,\
    # -0.01-0.1j, -0.01+0.1j, -0.01-0.1j, -0.01+0.1j, -0.01-0.1j])

    # poles_req = np.array([-0.03+0.1j, 0.115, -0.03-0.1j, -0.01+0.1j,\
    # -0.008-0.1j, -0.008+0.1j, -0.01-0.1j, -0.01+0.1j, -0.01-0.1j])
    # poles_req = np.array([-0.01+0.1j, 0.03, -0.01-0.1j, -0.01+0.115j,\
    # -0.01-0.115j, -0.01+0.1j, -0.01-0.1j, -0.008+0.1j, -0.008-0.1j])

    # best so far
    poles_req = np.array([-0.0085+0.01j, -0.0085-0.01j, -0.0085-0.01j,
                          -0.0085+0.01j, -0.0085-0.01j, -0.0085+0.01j,
                          -0.0085+0.01j, 0.025, -0.0085-0.01j])

    fsf = signal.place_poles(A, B, A_eigenvals, rtol=1e-4, maxiter=50)
    K = fsf.gain_matrix

    new_eigenvals, new_V = linalg.eig((A-np.matmul(B, K)))

    # x_0 = np.zeros(A.shape[0])
    # 0.01745 is degree in radians. this is psi here
    # x_0 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.01745])
    x_0 = np.array([0., 0., 0., 0., 0., 0., 0.01745, 0.01745, 0.01745])
    # x_0 = np.array([0., 0., 0., 0., 0., 0., 0., 0.01745, 0.])

    newA = A-np.matmul(B, K)
    newB = np.matmul(B, K)
    newC = np.identity(newA.shape[0])
    newD = np.zeros(newB.shape)

    new_sys = signal.dlti(newA, newB, newC, newD, dt=0.1)

    # step the new response, from t0 = 0 to tf = 2s
    sim = signal.dstep(new_sys, x0=x_0, n=21)
    times = np.linspace(0, 2, 21)

    # build plots
    philist = []
    thetalist = []
    psilist = []
    for i in range(len(sim[1][0])):
        philist.append(sim[1][0][i][-3])
        thetalist.append(sim[1][0][i][-2])
        psilist.append(sim[1][0][i][-1])

    # plot the responses for the sim
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times,
                             y=philist,
                             mode='lines',
                             name='phi'))
    fig.add_trace(go.Scatter(x=times,
                             y=thetalist,
                             mode='lines',
                             name='theta'))
    fig.add_trace(go.Scatter(x=times,
                             y=psilist,
                             mode='lines',
                             name='psi'))

    fig.show()
