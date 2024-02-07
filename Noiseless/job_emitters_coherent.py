# IMPORTING THE MODULES

import subprocess
import sys
import os
#os.environ["OMP_NUM_THREADS"] = str(16)

import numpy as np
from numpy import abs, sqrt, log10, sin, cos, exp
np.set_printoptions(precision=16)
from math import pi, factorial
from numpy.random import rand
from numpy.linalg import norm
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg, csc_matrix
import time
from scipy.optimize import minimize
import pickle

from qonn_cobyla import *

# =========================================================================================================

# DEFINITION OF FUNCTIONS

# Save output data to your preferred directory

def save(output, simulation_parameters):

    layers = simulation_parameters[2]

    store_dir = '/mnt/netapp1/Store_CSIC/home/csic/qia/amd/qfi_cfi_emitters_emitters/'
    import pickle 
    name = 'emitters_coherent_N_threshold'
    with open(store_dir + '/params_p_'+name+'_layers={}.p'.format(layers), 'wb') as fp:
        pickle.dump(output[0], fp)

    with open(store_dir + '/cost_p_'+name+'_layers={}.p'.format(layers), 'wb') as fp:
        pickle.dump(output[1], fp)

    with open(store_dir + '/params_m_'+name+'_layers={}.p'.format(layers), 'wb') as fp:
        pickle.dump(output[2], fp)

    with open(store_dir + '/cost_m_'+name+'_layers={}.p'.format(layers), 'wb') as fp:
        pickle.dump(output[3], fp)

    return

def optimization(simulation_parameters, N_p_list, phi, delta, phi_delta, max_iter, conv_tol, options):

    N_e = simulation_parameters[0]
    N_c = simulation_parameters[1]
    layers_p = simulation_parameters[2]
    layers_m = simulation_parameters[2]

    # Storing lists
    params_p_list = []
    params_m_list = []
    cost_p_list = []
    cost_m_list = []

    # Initial parameters for preparation quantum circuit
    np.random.seed(1234)
    parameters_p = np.array(0.1*(rand(3*layers_p)-0.5), dtype=np.float64)

    # Initial parameters for measurement quantum circuit
    np.random.seed(1996)
    parameters_m = np.array(0.1*(rand(3*layers_m)-0.5), dtype=np.float64)

    # JC circuit
    print('======> Emitters (JC) circuit')
    start = time.time()
    for N_p in N_p_list:

        print('N_p = {:}'.format(N_p))
        setup = Setup(N_e, N_c, N_p)
        oc = JCMeasCircuit(setup, layers_p, layers_m, delta)

        # Initial state: coherent state (up to an N_p/2 threshold in each mode)
        alpha = sqrt(N_p/4)
        psi_coh = 0.0
        for n in range(int(N_p/2 + 1)):
            cavity = np.zeros(N_p + 1, dtype=np.complex128)
            cavity[n] = 1.0
            psi_coh += alpha**n/sqrt(float(factorial(n)))*cavity
        psi_coh = exp(-abs(alpha)**2/2) * psi_coh
        psi_coh = psi_coh / sqrt(np.real(psi_coh[np.newaxis, :].conj() @ psi_coh[:, np.newaxis]))[0][0]
        psi_0 = np.kron(psi_coh, psi_coh)

        # Initial state: emitters in ground state
        emitter = np.array([1, 0], dtype=np.complex128)
        emitters = 1.0
        for i in range(N_e):
            emitters = np.kron(emitters, emitter)

        # Total initial state
        psi_0 = np.kron(emitters, psi_0)

        # Preparation VQC
        res = minimize(oc.preparation_qfi, parameters_p, args=(phi, phi_delta, psi_0), method='COBYLA',
            tol=conv_tol, options=options)

        params_p_list.append(res['x'])
        cost_p_list.append(oc.preparation_qfi(res['x'], phi, phi_delta, psi_0))

        parameters_p = res['x']
        
        print('Preparation + encoding done!')

        # Preparation circuit evaluation
        psi = oc.eval_rho(res['x'], psi_0)
        psi, grad_psi = oc.eval_rho_interf_grad(phi, psi)

        print('Circuit evaluation done!')

        # Measurement VQC
        res = minimize(oc.measurement_cfi, parameters_m, args=(psi, grad_psi), method='COBYLA',
            tol=conv_tol, options=options)

        params_m_list.append(res['x'])
        cost_m_list.append(oc.measurement_cfi(res['x'], psi, grad_psi))

        parameters_m = res['x']

        print('Measurement done!')

        output = [params_p_list, cost_p_list, params_m_list, cost_m_list]

        save(output, simulation_parameters)

    print('t = {:} s'.format(time.time() - start))

    return

def execution(simulation_parameters):

    N_e = simulation_parameters[0]
    N_c = simulation_parameters[1]
    layers = simulation_parameters[2]

    # Validate input
    if not isinstance(N_e, int):
        raise TypeError("Sorry. 'N_e' must be an integer (number of qubits).")
    if not isinstance(N_c, int):
        raise ValueError("Sorry. 'N_c' must be an integer (number of layers).")
    if not isinstance(layers, int):
        raise ValueError("Sorry. 'layers' must be an integer (number of parameters).")

    N_p_list = np.arange(4, 104, 4) # Number of photons threshold

    # Interferometer parameters

    phi = [0, pi/3, 0] # Phase to be estimated
    delta = 1e-2 # Small phase difference to calculate quantum Fisher information
    phi_delta = [0, pi/3 + delta, 0]

    # Optimization parameters
    max_iter = 1000 # Maximum number of iterations
    conv_tol = 1e-10 # Convergence tolerance
    options = {'maxiter': max_iter}

    optimization(simulation_parameters, N_p_list, phi, delta, phi_delta, max_iter, conv_tol, options)

    return

# Function to run in a HPC

def run():

    # Get job id from SLURM.
    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    print('jobid = ', jobid)
    print(sys.argv)
    N_e = int(sys.argv[1]) # Number of emitters
    N_c = int(sys.argv[2]) # Number of cavities
    layers = int(jobid) # Number of circuit layers

    simulation_parameters = [N_e, N_c, layers]

    execution(simulation_parameters)
    
    print('end!')
    return

if __name__ == '__main__':
    start_time = time.time()
    run()
    print('--- %s seconds ---' % (time.time()-start_time))
