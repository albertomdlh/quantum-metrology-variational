import numpy as np
from numpy import sqrt
import scipy
from scipy.linalg import sqrtm
import scipy.sparse as sp
from scipy.sparse import linalg, csc_matrix, csr_matrix, coo_matrix
from math import pi

class Circuit:

    def __init__(self):
        self.operators = []
        self.operators_interf = []

    def eval_rho(self, parameters, rho):

        for Omega, (A, B) in zip(parameters, self.operators):
            rho = sp.linalg.expm_multiply(-1j * (Omega * A + B), rho)
        
        return rho
    
    def eval_rho_interf(self, parameters, rho):

        for Omega, (A, B) in zip(parameters, self.operators_interf):
            rho = sp.linalg.expm_multiply(-1j * (Omega * A + B), rho)
        
        return rho
    
    def eval_rho_interf_grad(self, parameters, rho):
        W = [rho]
        for Omega, (A, B) in zip(parameters, self.operators_interf):
            W = [ sp.linalg.expm_multiply(-1j * (Omega * A + B), w) for w in W]
            W.append(-1j * A @ W[0])

        psi_output = W[0]
        psi_grad = W[1:]

        return psi_output, psi_grad[1]
    
    def eval_rho_meas(self, parameters, rho):

        for Omega, (A, B) in zip(parameters, self.operators_meas):
            rho = sp.linalg.expm_multiply(-1j * (Omega * A + B), rho)
        
        return rho

class JCMeasCircuit(Circuit):

    def __init__(self, setup, layers_p, layers_m, delta):
        super().__init__
        self.layers_p = layers_p
        self.layers_m = layers_m
        self.setup = setup
        self.operators = self.layers_p*[(setup.H_t(), 0), (setup.H_e(), 0), (setup.H_int(), 0)]
        self.operators_interf = [(0*sp.eye(setup.d, dtype=np.complex128), setup.H_BS_sym_ec()),
         (setup.H_phi_ec(), 0),
         (0*sp.eye(setup.d, dtype=np.complex128), setup.H_BS_sym_ec())]
        self.basis_e = csc_matrix(np.eye(2**setup.N_e, dtype=np.complex128))
        self.basis_c = csc_matrix(np.eye(setup.d_c, dtype=np.complex128))
        self.delta = delta
        self.operators_meas = self.layers_m*[(setup.H_t(), 0), (setup.H_e(), 0), (setup.H_int(), 0)]

    def preparation_qfi(self, parameters, phi, phi_delta, psi):

        # Preparation VQC
        psi = self.eval_rho(parameters, psi)

        # Encoding (MZ interferometer)
        psi_phi = self.eval_rho_interf(phi, psi)
        psi_delta = self.eval_rho_interf(phi_delta, psi)

        # Evaluate QFI
        F = np.abs(psi_phi[np.newaxis, :].conj() @ psi_delta[:, np.newaxis])
        cost = 8*(1-F)/self.delta**2

        return - cost[0][0]
    
    def measurement_cfi(self, parameters, psi, grad_psi):

        # Measurement VQC
        psi = self.eval_rho_meas(parameters, psi)
        grad_psi = self.eval_rho_meas(parameters, grad_psi)

        # Density matrices
        rho = psi[:, np.newaxis] @ psi[np.newaxis, :].conj()
        grad_rho = grad_psi[:, np.newaxis] @ psi[np.newaxis, :].conj()
        grad_rho += grad_rho.transpose().conj()

        # Evaluate classical Fisher information
        I = np.sum(1/np.diag(rho + 1e-30) * np.diag(grad_rho)**2)

        return - np.real(I)

class KerrMeasCircuit(Circuit):

    def __init__(self, setup, layers_p, layers_m, delta):
        super().__init__
        self.layers_p = layers_p
        self.layers_m = layers_m
        self.setup = setup
        self.operators = self.layers_p*[(setup.H_BS(), 0), (setup.H_Kerr(), 0)]
        self.operators_interf = [(0*sp.eye(setup.d_c, dtype=np.complex128), setup.H_BS_sym()),
         (setup.H_phi(), 0),
         (0*sp.eye(setup.d_c, dtype=np.complex128), setup.H_BS_sym())]
        self.delta = delta
        self.operators_meas = self.layers_m*[(setup.H_BS(), 0), (setup.H_Kerr(), 0)]

    def preparation_qfi(self, parameters, phi, phi_delta, psi):

        # Preparation VQC
        psi = self.eval_rho(parameters, psi)

        # Encoding (MZ interferometer)
        psi_phi = self.eval_rho_interf(phi, psi)
        psi_delta = self.eval_rho_interf(phi_delta, psi)

        # Evaluate QFI
        F = np.abs(psi_phi[np.newaxis, :].conj() @ psi_delta[:, np.newaxis])
        cost = 8*(1-F)/self.delta**2

        return - cost[0][0]
    
    def measurement_cfi(self, parameters, psi, grad_psi):

        # Measurement VQC
        psi = self.eval_rho_meas(parameters, psi)
        grad_psi = self.eval_rho_meas(parameters, grad_psi)

        # Density matrices
        rho = psi[:, np.newaxis] @ psi[np.newaxis, :].conj()
        grad_rho = grad_psi[:, np.newaxis] @ psi[np.newaxis, :].conj()
        grad_rho += grad_rho.transpose().conj()

        # Evaluate classical Fisher information
        I = np.sum(1/np.diag(rho + 1e-30) * np.diag(grad_rho)**2)

        return - np.real(I)



class Setup:

    def __init__(self, N_e, N_c, N_p):
        self.N_e = N_e # Number of emitters
        self.N_c = N_c # Number of cavities
        self.N_p = N_p # Photon threshold per cavity
        self.d_e = 2**self.N_e
        self.d_c = (self.N_p+1)**self.N_c # Cavities Hilbert space dimension
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

        # Tunneling
        H_t = 0
        for i in range(self.N_c):
            for j in range(i+1, self.N_c):
                H_t += self.a_i(i).H @ self.a_i(j)
        H_t += H_t.H

        return sp.kron(self.I_ec(), H_t) - sp.kron(H_t.conj(), self.I_ec())
    
    def H_e(self):

        # Emitters
        H_e = 0.0
        for i in range(self.N_e):
            H_e += self.Sm_i(i).H @ self.Sm_i(i)

        return H_e
    
    def H_e_vec(self):

        # Emitters
        H_e = 0.0
        for i in range(self.N_e):
            H_e += self.Sm_i(i).H @ self.Sm_i(i)

        return sp.kron(self.I_ec(), H_e) - sp.kron(H_e.conj(), self.I_ec())

    def H_c(self):

        # Cavities
        H_c = 0
        for i in range(self.N_c):
            H_c += self.a_i(i).H @ self.a_i(i)

        return H_c
    
    def H_c_vec(self):

        # Cavities
        H_c = 0
        for i in range(self.N_c):
            H_c += self.a_i(i).H @ self.a_i(i)

        return sp.kron(self.I_ec(), H_c) - sp.kron(H_c.conj(), self.I_ec())

    def H_int(self):

        # Interaction
        H_int = 0
        for i in range(self.N_e):
            H_int += self.Sm_i(i) @ self.a_i(i).H + self.Sm_i(i).H @ self.a_i(i)

        return H_int
    
    def H_int_vec(self):

        # Interaction
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
        Generator of the two-modes beamsplitter for vectorized density matrix.
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
        Generator of the two-modes beamsplitter (emitters + cavity Hilbert space).
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
        Generator of the two-modes symmetric beamsplitter for vectorized density matrix.
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
        Generator of the two-modes symmetric beamsplitter (emitters + cavity Hilbert space).
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
        Generator of the phase shift (emitters + cavity Hilbert space).
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
        'Generator of the Kerr interaction in the joint emitters + cavities Hilbert space.'

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
    
    def finite_zeros_c(self):

        basis = []
        for j in range(self.n+1):
            cavity_1 = np.zeros(self.N_p+1, dtype=np.complex128)
            cavity_1[j] = 1   
            for k in range(self.n+1-j):
                cavity_2 = np.zeros(self.N_p+1, dtype=np.complex128)
                cavity_2[k] = 1
                cavities = np.kron(cavity_1, cavity_2)
                basis.append(cavities)
        basis = np.array(basis)

        projector = 0
        for i in range(len(basis)):
            vec = basis[i]
            projector += vec[:, np.newaxis] @ vec[np.newaxis, :]

        P_sum = np.sum(projector, axis=0)
        P_finite = np.where(np.abs(P_sum) > 0)[0]
        P_zeros = np.where(P_sum == 0)[0]

        return P_finite, P_zeros
    
    def finite_zeros_ec(self):

        basis = []
        for j in range(self.n+1):
            cavity_1 = np.zeros(self.N_p+1, dtype=np.complex128)
            cavity_1[j] = 1   
            for k in range(self.n+1-j):
                cavity_2 = np.zeros(self.N_p+1, dtype=np.complex128)
                cavity_2[k] = 1
                cavities = np.kron(cavity_1, cavity_2)
                basis.append(cavities)
        basis = np.array(basis)

        projector = 0
        for i in range(len(basis)):
            vec = basis[i]
            projector += vec[:, np.newaxis] @ vec[np.newaxis, :]

        P_ec = np.kron(np.eye(2**self.N_e, dtype=np.complex128), projector)

        P_sum = np.sum(P_ec, axis=0)
        P_finite = np.where(np.abs(P_sum) > 0)[0]
        P_zeros = np.where(P_sum == 0)[0]

        return P_finite, P_zeros
    
    def P_c(self, operator):
        '''
        Reduces the number of degrees of freedom of the Hilbert sub-space of
        cavities to account for the conservation of the number of excitations.
        '''

        basis = []
        for j in range(self.n+1):
            cavity_1 = np.zeros(self.N_p+1, dtype=np.complex128)
            cavity_1[j] = 1   
            for k in range(self.n+1-j):
                cavity_2 = np.zeros(self.N_p+1, dtype=np.complex128)
                cavity_2[k] = 1
                cavities = np.kron(cavity_1, cavity_2)
                basis.append(cavities)
        basis = np.array(basis)

        projector = 0
        for i in range(len(basis)):
            vec = basis[i]
            projector += vec[:, np.newaxis] @ vec[np.newaxis, :]

        P_sum = np.sum(projector, axis=0)
        P_zeros = np.where(P_sum == 0)[0]


        #operator = operator.toarray()
        count = 0
        for i in P_zeros:
            operator = np.delete(operator, i-count, axis=0)
            operator = np.delete(operator, i-count, axis=1)
            count += 1

        #return csc_matrix(operator)
        return operator

    def P_ec(self, operator):
        '''
        Reduces the number of degrees of freedom of the total Hilbert space of
        emitters + cavities to account for the conservation of the number of excitations.
        '''

        basis = []
        for j in range(self.n+1):
            cavity_1 = np.zeros(self.N_p+1, dtype=np.complex128)
            cavity_1[j] = 1   
            for k in range(self.n+1-j):
                cavity_2 = np.zeros(self.N_p+1, dtype=np.complex128)
                cavity_2[k] = 1
                cavities = np.kron(cavity_1, cavity_2)
                basis.append(cavities)
        basis = np.array(basis)

        projector = 0
        for i in range(len(basis)):
            vec = basis[i]
            projector += vec[:, np.newaxis] @ vec[np.newaxis, :]

        P_ec = np.kron(np.eye(2**self.N_e, dtype=np.complex128), projector)

        P_sum = np.sum(P_ec, axis=0)
        P_zeros = np.where(P_sum == 0)[0]


        #operator = operator.toarray()
        count = 0
        for i in P_zeros:
            operator = np.delete(operator, i-count, axis=0)
            operator = np.delete(operator, i-count, axis=1)
            count += 1

        #return csc_matrix(operator)
        return operator


