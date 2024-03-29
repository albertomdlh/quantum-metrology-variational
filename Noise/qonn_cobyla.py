# We import the modules

import numpy as np
from numpy import sqrt
import scipy
from scipy.linalg import sqrtm
import scipy.sparse as sp
from scipy.sparse import linalg, csc_matrix, csr_matrix, coo_matrix
from math import pi

# Superclass representing a general variational quantum circuit

class Circuit:

    def __init__(self):
        self.operators = []
        self.operators_interf = []

    def eval_rho(self, parameters, rho):
        '''
        This function applies the unitaries of the preparation quantum circuit
        to the input state of the algorithm.
        "parameters" is a list of parameters, each multiplied to the generators
        contained in the list "self.operators". "rho" is a quantum state.
        '''

        for Omega, (A, B) in zip(parameters, self.operators):
            rho = sp.linalg.expm_multiply(-1j * (Omega * A + B), rho)
        
        return rho
    
    def eval_rho_interf(self, parameters, rho):
        '''
        This function applies the unitaries of the interferometer
        to the output state of the preparation VQC.
        "parameters" is a list of parameters, each multiplied to the generators
        contained in the list "self.operators". "rho" is a quantum state.
        '''

        for Omega, (A, B) in zip(parameters, self.operators_interf):
            rho = sp.linalg.expm_multiply(-1j * (Omega * A + B), rho)
        
        return rho
    
    def eval_rho_interf_grad(self, parameters, rho):
        '''
        This function applies the unitaries of the interferometer
        to the output state of the preparation VQC, and also returns the gradient
        with respect to the encoded phase.
        "parameters" is a list of parameters, each multiplied to the generators
        contained in the list "self.operators". "rho" is a quantum state.
        '''

        W = [rho]
        for Omega, (A, B) in zip(parameters, self.operators_interf):
            W = [ sp.linalg.expm_multiply(-1j * (Omega * A + B), w) for w in W]
            W.append(-1j * A @ W[0])

        psi_output = W[0]
        psi_grad = W[1:]

        return psi_output, psi_grad[1]
    
    def eval_rho_meas(self, parameters, rho):
        '''
        This function applies the unitaries of the preparation quantum circuit
        to the output state of the phase encoding process.
        "parameters" is a list of parameters, each multiplied to the generators
        contained in the list "self.operators". "rho" is a quantum state.
        '''

        for Omega, (A, B) in zip(parameters, self.operators_meas):
            rho = sp.linalg.expm_multiply(-1j * (Omega * A + B), rho)
        
        return rho
    
# We now particularize the superclass above for the emitters (Jaynes-Cummings like) variational quantum circuit

class JCMeasCircuit(Circuit):

    def __init__(self, setup, layers_p, layers_m, delta, kappa):
        super().__init__
        self.layers_p = layers_p # Number of preparation layers
        self.layers_m = layers_m # Number of measurement layers
        self.setup = setup # Loads setup class with all predifined quantum gates and generators
        self.operators = self.layers_p*[(setup.H_t(), 0), (setup.H_e(), 0), (setup.H_int(), 0)] # Operators for preparation VQC
        self.operators_interf = [(0*sp.eye(setup.d, dtype=np.complex128), setup.H_BS_sym_ec()),
         (setup.H_phi_ec(), 0),
         (0*sp.eye(setup.d, dtype=np.complex128), setup.H_BS_sym_ec())] # Operators for interferometer (phase encoding)
        self.basis_e = csc_matrix(np.eye(2**setup.N_e, dtype=np.complex128)) # Emitters basis for tracing out degrees of freedom and calculating reduced density matrices
        self.basis_c = csc_matrix(np.eye(setup.d_c, dtype=np.complex128))  # Cavity (photons) basis for tracing out degrees of freedom and calculating reduced density matrices
        self.delta = delta # Small parameter to calculate quantum Fisher information
        self.operators_meas = self.layers_m*[(setup.H_t_vec(), 0), (setup.H_e_vec(), 0), (setup.H_int_vec(), 0)] # Operators for measurement VQC
        
        # Lindbladian for amplitude damping noise (to be applied to vectorized density matrices)
        L_ad = 0.0
        for k in range(setup.N_c):
            L_ad += kappa*(sp.kron(setup.a_i(k).conj(), setup.a_i(k)) - 0.5*sp.kron(sp.eye(setup.d, dtype=np.complex128),
                setup.a_i(k).H @ setup.a_i(k)) - 0.5*sp.kron((setup.a_i(k).H @ setup.a_i(k)).transpose(), sp.eye(setup.d, dtype=np.complex128)))
        self.L_ad = L_ad

        # Lindbladian for phase damping noise (to be applied to vectorized density matrices)
        L_pd = 0.0
        for k in range(setup.N_c):
            L_pd += kappa*(sp.kron(setup.n_i(k).conj(), setup.n_i(k)) - 0.5*sp.kron(sp.eye(setup.d, dtype=np.complex128),
                setup.n_i(k).H @ setup.n_i(k)) - 0.5*sp.kron((setup.n_i(k).H @ setup.n_i(k)).transpose(), sp.eye(setup.d, dtype=np.complex128)))
        self.L_pd = L_pd

    def preparation_qfi(self, parameters, phi, phi_delta, psi):

        # Preparation VQC
        psi = self.eval_rho(parameters, psi)

        # Encoding (MZ interferometer)
        psi_phi = self.eval_rho_interf(phi, psi)
        psi_delta = self.eval_rho_interf(phi_delta, psi)

        # Density matrices
        rho_phi = psi_phi[:, np.newaxis] @ psi_phi[np.newaxis, :].conj()
        rho_delta = psi_delta[:, np.newaxis] @ psi_delta[np.newaxis, :].conj()

        # Vectorize density matrices
        rho_phi = np.reshape(rho_phi, (len(rho_phi)**2, 1))
        rho_delta = np.reshape(rho_delta, (len(rho_delta)**2, 1))

        # Losses
        rho_phi = sp.linalg.expm_multiply(self.L_ad, rho_phi)
        rho_delta = sp.linalg.expm_multiply(self.L_ad, rho_delta)
        rho_phi = sp.linalg.expm_multiply(self.L_pd, rho_phi)
        rho_delta = sp.linalg.expm_multiply(self.L_pd, rho_delta)
        rho_phi = np.reshape(rho_phi, (int(sqrt(len(rho_phi))), int(sqrt(len(rho_phi)))))
        rho_delta = np.reshape(rho_delta, (int(sqrt(len(rho_delta))), int(sqrt(len(rho_delta))))) 

        # Evaluate QFI
        F = np.real(np.trace(sqrtm(sqrtm(rho_phi) @ rho_delta @ sqrtm(rho_phi))))
        cost = 8*(1-F)/self.delta**2

        return - cost
    
    def measurement_cfi(self, parameters, rho, grad_rho):

        # Measurement VQC
        rho = self.eval_rho_meas(parameters, rho)
        grad_rho = self.eval_rho_meas(parameters, grad_rho)

        # Reconstruct density matrices
        rho = np.reshape(rho, (int(sqrt(len(rho))), int(sqrt(len(rho)))))
        grad_rho = np.reshape(grad_rho, (int(sqrt(len(grad_rho))), int(sqrt(len(grad_rho))))) 

        # Evaluate classical Fisher information
        I = np.sum(1/np.diag(rho + 1e-30) * np.diag(grad_rho)**2)

        return - np.real(I)
    
# We now particularize the superclass above for the Kerr non-linearity variational quantum circuit

class KerrMeasCircuit(Circuit):

    def __init__(self, setup, layers_p, layers_m, delta, kappa):
        super().__init__
        self.layers_p = layers_p # Number of preparation layers
        self.layers_m = layers_m # Number of measurement layers
        self.setup = setup # Loads setup class with all predifined quantum gates and generators
        self.operators = self.layers_p*[(setup.H_BS(), 0), (setup.H_Kerr(), 0)] # Operators for preparation VQC
        self.operators_interf = [(0*sp.eye(setup.d_c, dtype=np.complex128), setup.H_BS_sym()),
         (setup.H_phi(), 0),
         (0*sp.eye(setup.d_c, dtype=np.complex128), setup.H_BS_sym())] # Operators for interferometer (phase encoding)
        self.delta = delta # Small parameter to calculate quantum Fisher information
        self.operators_meas = self.layers_m*[(setup.H_BS_vec(), 0), (setup.H_Kerr_vec(), 0)] # Operators for measurement VQC
        
        # Lindbladian for amplitude damping noise (to be applied to vectorized density matrices)
        L_ad = 0.0
        for k in range(setup.N_c):
            L_ad += kappa*(sp.kron(setup.a_c(k).conj(), setup.a_c(k)) - 0.5*sp.kron(sp.eye(setup.d_c, dtype=np.complex128),
                setup.a_c(k).H @ setup.a_c(k)) - 0.5*sp.kron((setup.a_c(k).H @ setup.a_c(k)).transpose(), sp.eye(setup.d_c, dtype=np.complex128)))
        self.L_ad = L_ad

        # Lindbladian for phase damping noise (to be applied to vectorized density matrices)
        L_pd = 0.0
        for k in range(setup.N_c):
            L_pd += kappa*(sp.kron(setup.n_c(k).conj(), setup.n_c(k)) - 0.5*sp.kron(sp.eye(setup.d_c, dtype=np.complex128),
                setup.n_c(k).H @ setup.n_c(k)) - 0.5*sp.kron((setup.n_c(k).H @ setup.n_c(k)).transpose(), sp.eye(setup.d_c, dtype=np.complex128)))
        self.L_pd = L_pd

    def preparation_qfi(self, parameters, phi, phi_delta, psi):

        # Preparation VQC
        psi = self.eval_rho(parameters, psi)

        # Encoding (MZ interferometer)
        psi_phi = self.eval_rho_interf(phi, psi)
        psi_delta = self.eval_rho_interf(phi_delta, psi)

        # Density matrices
        rho_phi = psi_phi[:, np.newaxis] @ psi_phi[np.newaxis, :].conj()
        rho_delta = psi_delta[:, np.newaxis] @ psi_delta[np.newaxis, :].conj()

        # Vectorize density matrices
        rho_phi = np.reshape(rho_phi, (len(rho_phi)**2, 1))
        rho_delta = np.reshape(rho_delta, (len(rho_delta)**2, 1))

        # Losses
        rho_phi = sp.linalg.expm_multiply(self.L_ad, rho_phi)
        rho_delta = sp.linalg.expm_multiply(self.L_ad, rho_delta)
        rho_phi = sp.linalg.expm_multiply(self.L_pd, rho_phi)
        rho_delta = sp.linalg.expm_multiply(self.L_pd, rho_delta)
        rho_phi = np.reshape(rho_phi, (int(sqrt(len(rho_phi))), int(sqrt(len(rho_phi)))))
        rho_delta = np.reshape(rho_delta, (int(sqrt(len(rho_delta))), int(sqrt(len(rho_delta))))) 

        # Evaluate QFI
        F = np.real(np.trace(sqrtm(sqrtm(rho_phi) @ rho_delta @ sqrtm(rho_phi))))
        cost = 8*(1-F)/self.delta**2

        return - cost
    
    def measurement_cfi(self, parameters, rho, grad_rho):

        # Measurement VQC
        rho = self.eval_rho_meas(parameters, rho)
        grad_rho = self.eval_rho_meas(parameters, grad_rho)

        # Reconstruct density matrices
        rho = np.reshape(rho, (int(sqrt(len(rho))), int(sqrt(len(rho)))))
        grad_rho = np.reshape(grad_rho, (int(sqrt(len(grad_rho))), int(sqrt(len(grad_rho))))) 

        # Evaluate classical Fisher information
        I = np.sum(1/np.diag(rho + 1e-30) * np.diag(grad_rho)**2)

        return - np.real(I)

# The class below contains all necessary quantum gates and generators

class Setup:

    def __init__(self, N_e, N_c, N_p):
        self.N_e = N_e # Number of emitters
        self.N_c = N_c # Number of cavities
        self.N_p = N_p # # Number of photons threshold
        self.d_e = 2**self.N_e # Emitters Hilbert space dimension
        self.d_c = (self.N_p+1)**self.N_c # Cavities (photons) Hilbert space dimension
        self.d = 2**self.N_e*(self.N_p+1)**self.N_c # Total Hilbert space dimension

    def Sm(self):
        '''Local sigma_- operator'''
        Sm = csc_matrix([
            [0, 1],
            [0, 0]
            ], dtype=np.complex128)
        return Sm

    def X(self):
        '''Local sigma_x operator'''
        Sx = csc_matrix([
            [0, 1],
            [1, 0]
            ], dtype=np.complex128)
        return Sx

    def Y(self):
        '''Local sigma_y operator'''
        Sy = csc_matrix([
            [0, -1j],
            [1j, 0]
            ], dtype=np.complex128)
        return Sy

    def Z(self):
        '''Local sigma_z operator'''
        Sz = csc_matrix([
            [1, 0],
            [0, -1]
            ], dtype=np.complex128)
        return Sz
    
    def H(self):
        '''Single-qubit Hadamard gate'''
        H = 1/sqrt(2)*csc_matrix([
            [1, 1],
            [1, -1]
            ], dtype=np.complex128)
        return H

    def a(self):
        '''
        Local annihilation operator acting in the Hilbert space of a single cavity mode.
        ...

        Inputs:
        -------
        psi : photon threshold

        '''

        matrix = np.zeros((self.N_p+1, self.N_p+1), dtype=np.complex128)
        for i in range(self.N_p):
                matrix[i][i+1] = sqrt(i+1)
        return csc_matrix(matrix, dtype=np.complex128)

    def local_operator_emitters(self, operator, i):
        '''
        Creates an operator 1 \otimes ... operator \otimes 1
        where operator acts on the ith emitter.
        ...

        Parameters
        -------
        operator : sparse matrix
            The operator acting locally on a single emitter.
        i : int
            The ith emitter in which operator acts. The first is i=0.
        '''

        operator_ec = sp.kron(sp.eye(2**i, dtype=np.complex128),
                        sp.kron(operator, sp.eye(2**(self.N_e-(i+1))*(self.N_p+1)**self.N_c, dtype=np.complex128)))
        
        return operator_ec

    def local_operator_cavity(self, operator, i):
        '''
        Creates an operator 1 \otimes ... operator \otimes 1
        where operator acts on the ith cavity mode.
        ...

        Parameters
        -------
        operator : numpy array
            The operator acting locally on a single cavity mode.
        i : int
            The ith cavity mode in which operator acts. The first is i=0.
        '''

        operator_ec = sp.kron(sp.eye(2**self.N_e*(self.N_p+1)**i, dtype=np.complex128),
         sp.kron(operator, sp.eye((self.N_p+1)**(self.N_c-(i+1)), dtype=np.complex128)))

        return operator_ec


    def Sm_i(self, index):
        '''
        sigma_- operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator_emitters(self.Sm(), index)

    def X_i(self, index):
        '''
        X operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator_emitters(self.X(), index)

    def Y_i(self, index):
        '''
        Y operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator_emitters(self.Y(), index)

    def Z_i(self, index):
        '''
        Z operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator_emitters(self.Z(), index)
    
    def H_i(self, index):
        '''
        Hadamard gate acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which the gate acts.
        '''

        return self.local_operator_emitters(self.H(), index)

    def a_i(self, index):
        '''
        Local cavity annihilation operator acting on cavity i in the
        total emitters + cavity Hilbert space.
        '''

        return self.local_operator_cavity(self.a(), index)
    
    def a_c(self, i):
        '''
        Local cavity annihilation operator acting on cavity i in the
        cavities Hilbert subspace.
        '''

        operator_c = sp.kron(sp.eye((self.N_p+1)**i, dtype=np.complex128),
         sp.kron(self.a(), sp.eye((self.N_p+1)**(self.N_c-i-1), dtype=np.complex128)))
    
        return operator_c
    
    def n_c(self, i):
        '''
        Local cavity number operator acting on cavity i in the
        cavities Hilbert subspace.
        '''

        n = self.a_c(i).H @ self.a_c(i)

        return n 
    
    def n_i(self, i):
        '''
        Local cavity number operator acting on cavity i in the
        total cavities + emitters Hilbert space.
        '''

        n = self.a_i(i).H @ self.a_i(i)

        return n 
    
    def I_c(self):
        '''
        Identity matrix for the cavities Hilbert space.
        '''
        
        id = sp.eye(self.d_c)

        return id
    
    def I_ec(self):
        '''
        Identity matrix for the emitters + cavities Hilbert space.
        '''
        
        id = sp.eye(self.d)

        return id

    def H_t(self):

        # Tunneling
        H_t = 0
        for i in range(self.N_c):
            for j in range(i+1, self.N_c):
                H_t += self.a_i(i).H @ self.a_i(j)
        H_t += H_t.H

        return H_t
    
    def H_t_vec(self):

        # Tunneling vectorized to use with density matrices living in the emitters + cavities Hilbert space
        H_t = 0
        for i in range(self.N_c):
            for j in range(i+1, self.N_c):
                H_t += self.a_i(i).H @ self.a_i(j)
        H_t += H_t.H

        return sp.kron(self.I_ec(), H_t) - sp.kron(H_t.conj(), self.I_ec())
    
    def H_e(self):

        # Emitters-cavity detuning
        H_e = 0.0
        for i in range(self.N_e):
            H_e += self.Sm_i(i).H @ self.Sm_i(i)

        return H_e
    
    def H_e_vec(self):

        # Emitters-cavity detuning vectorized to use with density matrices living in the emitters + cavities Hilbert space
        H_e = 0.0
        for i in range(self.N_e):
            H_e += self.Sm_i(i).H @ self.Sm_i(i)

        return sp.kron(self.I_ec(), H_e) - sp.kron(H_e.conj(), self.I_ec())

    def H_c(self):

        # Cavities mode frequency
        H_c = 0
        for i in range(self.N_c):
            H_c += self.a_i(i).H @ self.a_i(i)

        return H_c
    
    def H_c_vec(self):

        # Cavities mode frequency vectorized to use with density matrices living in the emitters + cavities Hilbert space
        H_c = 0
        for i in range(self.N_c):
            H_c += self.a_i(i).H @ self.a_i(i)

        return sp.kron(self.I_ec(), H_c) - sp.kron(H_c.conj(), self.I_ec())

    def H_int(self):

        # Emitters-photon interaction
        H_int = 0
        for i in range(self.N_e):
            H_int += self.Sm_i(i) @ self.a_i(i).H + self.Sm_i(i).H @ self.a_i(i)

        return H_int
    
    def H_int_vec(self):

        # Emitters-photon interaction vectorized to use with density matrices living in the emitters + cavities Hilbert space
        H_int = 0
        for i in range(self.N_e):
            H_int += self.Sm_i(i) @ self.a_i(i).H + self.Sm_i(i).H @ self.a_i(i)

        return sp.kron(self.I_ec(), H_int) - sp.kron(H_int.conj(), self.I_ec())
    
    def H_BS(self):
        '''
        Generator of the two-modes beamsplitter.
        '''

        # 2 modes only
        H = self.a_c(0).H @ self.a_c(1) + self.a_c(0) @ self.a_c(1).H
    
        return H
    
    def H_BS_vec(self):
        '''
        Generator of the two-modes beamsplitter for vectorized density matrix (cavity Hilbert space).
        '''

        # 2 modes only
        H = self.a_c(0).H @ self.a_c(1) + self.a_c(0) @ self.a_c(1).H
    
        return sp.kron(self.I_c(), H) - sp.kron(H.conj(), self.I_c())
    
    def H_BS_ec(self):
        '''
        Generator of the two-modes beamsplitter (emitters + cavity Hilbert space).
        '''

        # 2 modes only
        H = self.a_i(0).H @ self.a_i(1) + self.a_i(0) @ self.a_i(1).H
    
        return H
    
    def H_BS_ec_vec(self):
        '''
        Generator of the two-modes beamsplitter for vectorized density matrices (emitters + cavity Hilbert space).
        '''

        # 2 modes only
        H = self.a_i(0).H @ self.a_i(1) + self.a_i(0) @ self.a_i(1).H
    
        return sp.kron(self.I_ec(), H) - sp.kron(H.conj(), self.I_ec())

    def H_BS_sym(self):
        '''
        Generator of the two-modes symmetric beamsplitter.
        '''

        # 2 modes only
        H = -pi/4*(self.a_c(0).H @ self.a_c(1) + self.a_c(0) @ self.a_c(1).H)
    
        return H
    
    def H_BS_sym_vec(self):
        '''
        Generator of the two-modes symmetric beamsplitter for vectorized density matrix (cavity Hilbert space).
        '''

        # 2 modes only
        H = -pi/4*(self.a_c(0).H @ self.a_c(1) + self.a_c(0) @ self.a_c(1).H)
    
        return sp.kron(self.I_c(), H) - sp.kron(H.conj(), self.I_c())
    
    def H_BS_sym_ec(self):
        '''
        Generator of the two-modes symmetric beamsplitter (emitters + cavity Hilbert space).
        '''

        # 2 modes only
        H = -pi/4*(self.a_i(0).H @ self.a_i(1) + self.a_i(0) @ self.a_i(1).H)
    
        return H
    
    def H_BS_sym_ec_vec(self):
        '''
        Generator of the two-modes symmetric beamsplitter for vectorized density matrices (emitters + cavity Hilbert space).
        '''

        # 2 modes only
        H = -pi/4*(self.a_i(0).H @ self.a_i(1) + self.a_i(0) @ self.a_i(1).H)
    
        return sp.kron(self.I_ec(), H) - sp.kron(H.conj(), self.I_ec())
    
    def H_phi(self):
        '''
        Generator of the phase shift.
        '''

        # 2 modes only
        H = self.a_c(1).H @ self.a_c(1)

        return H
    
    def H_phi_vec(self):
        '''
        Generator of the phase shift for vectorized density matrix.
        '''

        # 2 modes only
        H = self.a_c(1).H @ self.a_c(1)

        return sp.kron(self.I_c(), H) - sp.kron(H.conj(), self.I_c())

    def H_phi_ec(self):
        '''
        Generator of the phase shift (emitters + cavity Hilbert space).
        '''

        # 2 modes only
        H = self.a_i(1).H @ self.a_i(1)

        return H
    
    def H_phi_ec_vec(self):
        '''
        Generator of the phase shift for vectorized density matrices (emitters + cavity Hilbert space).
        '''

        # 2 modes only
        H = self.a_i(1).H @ self.a_i(1)

        return sp.kron(self.I_ec(), H) - sp.kron(H.conj(), self.I_ec())

    def H_Kerr(self):
        'Generator of the Kerr interaction.'

        H = 0
        for i in range(self.N_c):
            H += (self.a_c(i).H @ self.a_c(i))**2 - self.a_c(i).H @ self.a_c(i)

        return H
    
    def H_Kerr_vec(self):
        'Generator of the Kerr interaction for vectorized density matrix.'

        H = 0
        for i in range(self.N_c):
            H += (self.a_c(i).H @ self.a_c(i))**2 - self.a_c(i).H @ self.a_c(i)

        return sp.kron(self.I_c(), H) - sp.kron(H.conj(), self.I_c())
    
    def H_Kerr_ec(self):
        'Generator of the Kerr interaction in the joint emitters + cavities Hilbert space.'

        H = 0
        for i in range(self.N_c):
            H += (self.a_c(i).H @ self.a_c(i))**2 - self.a_c(i).H @ self.a_c(i)

        return H
    
    def H_Kerr_ec_vec(self):
        'Generator of the Kerr interaction for vectorized density matrices in the joint emitters + cavities Hilbert space.'

        H = 0
        for i in range(self.N_c):
            H += (self.a_c(i).H @ self.a_c(i))**2 - self.a_c(i).H @ self.a_c(i)

        return sp.kron(self.I_ec(), H) - sp.kron(H.conj(), self.I_ec())
    
    def H_rot(self):

        # Rotation
        H = 0.0
        for i in range(self.N_e):
            H += self.X_i(i)

        return 0.5*H
    
    def L_emitters_ad(self):
        '''
        Lindbladian for amplitude damping in the cavity + emitters Hilbert space.
        To be applied to vectorized density matrices.
        '''
    
        L = 0.0
        for k in range(self.N_c):
            L += sp.kron(self.a_i(k).conj(), self.a_i(k)) - 0.5*sp.kron(sp.eye(self.d, dtype=np.complex128),
                self.a_i(k).H @ self.a_i(k)) - 0.5*sp.kron((self.a_i(k).H @ self.a_i(k)).transpose(), sp.eye(self.d, dtype=np.complex128))

        return L
    
    def L_emitters_pd(self):
        '''
        Lindbladian for phase damping in the cavity + emitters Hilbert space.
        To be applied to vectorized density matrices.
        '''

        L = 0.0
        for k in range(self.N_c):
            L += sp.kron(self.n_i(k).conj(), self.n_i(k)) - 0.5*sp.kron(sp.eye(self.d, dtype=np.complex128),
                self.n_i(k).H @ self.n_i(k)) - 0.5*sp.kron((self.n_i(k).H @ self.n_i(k)).transpose(), sp.eye(self.d, dtype=np.complex128))

        return L
    
    def L_Kerr_ad(self):
        '''
        Lindbladian for amplitude damping in the cavity (only photons) Hilbert space.
        To be applied to vectorized density matrices.
        '''
    
        L = 0.0
        for k in range(self.N_c):
            L += sp.kron(self.a_c(k).conj(), self.a_c(k)) - 0.5*sp.kron(sp.eye(self.d_c, dtype=np.complex128),
                self.a_c(k).H @ self.a_c(k)) - 0.5*sp.kron((self.a_c(k).H @ self.a_c(k)).transpose(), sp.eye(self.d_c, dtype=np.complex128))

        return L
    
    def L_Kerr_pd(self):
        '''
        Lindbladian for phase damping in the cavity (only photons) Hilbert space.
        To be applied to vectorized density matrices.
        '''
    
        L = 0.0
        for k in range(self.N_c):
            L += sp.kron(self.n_c(k).conj(), self.n_c(k)) - 0.5*sp.kron(sp.eye(self.d_c, dtype=np.complex128),
                self.n_c(k).H @ self.n_c(k)) - 0.5*sp.kron((self.n_c(k).H @ self.n_c(k)).transpose(), sp.eye(self.d_c, dtype=np.complex128))

        return L
    
    def U_MZ(self, phi):
        '''
        Unitary of the Mach-Zehnder interferometer.
        '''

        # 2 modes only
        U_BS = scipy.linalg.expm(1j*pi/4*(self.a_c(0).H.toarray() @ self.a_c(1).toarray() + self.a_c(0).toarray() @ self.a_c(1).H.toarray()))
        #U_phi = scipy.linalg.expm(-1j*phi/2*(self.a_c(0).H.toarray() @ self.a_c(0).toarray() - self.a_c(1).H.toarray() @ self.a_c(1).toarray()))
        U_phi = scipy.linalg.expm(-1j*phi/2*self.a_c(1).H.toarray() @ self.a_c(1).toarray())

        return U_BS @ U_phi @ U_BS
    
    def H_D_i(self, i):
        'Displacement generator applied to cavity mode i.'
        H = self.a_i(i).H - self.a_i(i)
        return H
    
    def H_D(self):
        'Layer of displacement generators applied to all cavity modes.'
        H = 0.0
        for i in range(self.N_c):
            H = self.H_D_i(i) + H
        return H
    
    def H_S_i(self, i):
        'Squeezing generator applied to cavity mode i.'
        H = 0.5*(self.a_i(i)**2 - self.a_i(i).H**2)
        return H

    def H_S(self):
        'Layer of squeezing generators applied to all cavity modes.'
        H = 0.0
        for i in range(self.N_c):
            H = self.H_S_i(i) + H
        return H
    
