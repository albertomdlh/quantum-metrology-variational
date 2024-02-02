# quantum-metrology-variational
This GitHub repository contains the necessary codes (written in Python) to reproduce the figures of the paper "Photonic quantum metrology with variational quantum optical non-linearities".

The repository contains two folders:

1) Noiseless: the codes contained here do not take into account the presence of noise in the interferometric setup. These codes can be used to reproduce Figures 2 and 3 of the manuscript. In this folder we can find the following files:
1.1.- "job_emitters_coherent.py": employs the emitters ansatz starting from two identical coherent states.
1.2.- "job_qfi_cfi_kerr.py": employs the Kerr non-linearity ansatz starting from two identical coherent states. It includes the possibility to restrict the Kerr non-linearity strength using a bound.
1.3.- "qonn_cobyla.py": the library containing all functions employed by the codes above.

3) Noise: here we take into account noise (amplitude damping, i.e., photon loss, and phase damping, i.e., decoherence) in the interferometric setup. These codes can be used to reproduce Figure 4 of the manuscript. In this folder we can find the following files:
2.1.- "job_emitters_coherent_asaf_kappa.py": employs the emitters ansatz starting from two identical coherent states.
2.2.- "job_kerr_coherent_unbound_asaf_kappa.py": employs the Kerr non-linearity ansatz starting from two identical coherent states.
2.3.- "qonn_cobyla.py": the library containing all functions employed by the codes above.

