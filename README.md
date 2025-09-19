# Extended Hubbard Model

This repository is intended to provied a set of tools to create and manipulate parameter's dependent Extended Hubbard Models. These kind of models are defined as

```math
    \hat{H} = \sum_{j,k=1}^{N} t_{jk} c_{j}^{\dagger}c_k + \frac{1}{2} \sum_{h,j,k,l} U_{h,j,k,l} c_k^\dagger c_j^\dagger c_k c_m,
```

where $N$ is the number of single-particle states. $t_{jk} = < j|\hat{H}_0 |k>$ is obtained from the single-particle $N\times N$ hamiltonian and $U_{hjkm}=< hj |\hat{H}_C|km>$ captures the pairwise particle-particle interactions.

The main class which allows for creating this kind of Hamiltonians is ```ManyBodyHamiltonian```. To initialize it one has to provide the number of single-particle states (or orbitals) ```norb```, the number of fermionic particles which will be considered ```n_elec``` (note that more than 2 particles can be included but only up to particle-particle interactions can be modelled), a dictionary with the single-particle terms ```hdict``` and a dictionary with the particle-particle terms ```cdict```. 

Each term in ```hdict``` must be of the shape ```{i,j: value}``` with $i<=j$, corresponding to the single-particle Hamiltonian term $< i|H|j>$. Each term in ```cdict``` must be of the shape ´´´{(h,j,k,l): value}´´´ with $j>h$, corresponding to the interaction term $U_{h,j,k,l}$.

The low level operations are made with bit determinant. For example, the basis element corresponding to the first and third orbitals filled in an $N=4$ system is represented by the number 5, which in bit representation is $0b0101$ (read from right to left). All these low level operations are made by the class ```FockSpaceUtilities```.

```ManyBodyHamiltonian``` allows for obtain the eigenvalues spectrum of the Hamiltonian given, to change the base from the initial Fock Space base to any other, to perform a Schrieffer Wolf transformation using Pymablock and to create well defined decoherence operators for a dynamic simulation with qutip. All of these operations is agnostic of the meaning and labels of the single-particle states provided. Therefore, is it more useful to create a particular instance of the Hamiltonian with the corresponding basis labels. In this code this particularization is made for a Bilayer Graphene Double Quantum Dot with two electrons and one orbital per dot.


# DQD21

DQD21 allows to define a BLG DQD with 2 electrons and 1 orbital per electron using the model explained in "Extended Hubbard model describing small multidot arrays in bilayer graphene, Knothe, A. and Burkard, G. (2024)". The different parameters of the model can be inserted easily with a dictionary and the class will convert them to well suited dictionaries for the ```ManyBodyHamiltonian``` class. An important part of this class are the functions ```_initilizeGlobalLabels``` an ```createBases```, which define new basis to transform the Hamiltonian with understandable labels. The correspondence of each state with the states of the form $c_i^{\dagger}c_{j}^{\dagger}|0>$ has been done manually and insterted, just once, in ```_initilizeGlobalLabels```. Then, to create any ordering of this singlet-triplet basis, one just have to call to the labels as is it done in ```createBases```. This helps to project the Hamiltonian in different basis and to perform well informed Schrieffer Wolf Transformations.  


All the utilities of the constructions are better understood in the particular scripts. In the folder `spectrum_analysis` one can see how the Hamiltonian changes if we project or reorder the basis ("hamiltonian_heatmap.py") or the spectrum with respect to a certain parameter (usually detuning) and how similar is each eigenvalue to a particular basis state. 

In the folder `dynamics_analysis` a a parameter dependent protocol can be explored, using the class `DynamicsManager`. The script `plottingMethods` is intended to be as agnostic and reusable and possible, but most of the methods are constructed to serve the particularities of DQD21.