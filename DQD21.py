import numpy as np

from typing import List, Tuple, Optional
from enum import Enum
from qutip import Result, Qobj

from src.ManyBodyHamiltonian import ManyBodyHamiltonian
from src.DynamicsManager import DynamicsManager

from copy import deepcopy


class DQDParameters(Enum):
    B_Z = 'bz'
    B_X = 'bx'
    GS_L = 'gsLeft'
    GS_R = 'gsRight'
    GV_L = 'gvLeft'
    GV_R = 'gvRight'
    DELTA_SO = 'DeltaSO'
    DELTA_KK = 'DeltaKK'
    E_R = 'ERight'
    TUNNELING = 't'
    T_SOC = 't_soc'
    U0 = 'U0'
    U1 = 'U1'
    X = 'X'
    A = 'A'
    P = 'P'
    J = 'J'
    G_ORTHO = 'g_ortho'
    G_ZZ = 'g_zz'
    G_Z0 = 'g_z0'
    G_0Z = 'g_0z'

class BasesNames(Enum):
    FOCK_DETERMINANT = 0
    SINGLET_TRIPLET = 1
    SINGLET_TRIPLET_QUBIT_4 = 2
    SIMMETRIC_ANTISYMMETRIC = 3
    SINGLET_TRIPLET_QUBIT_VALLEY_MINUS = 4


class DQD21(ManyBodyHamiltonian, DynamicsManager):
    """
    Class to represent a double quantum dot with 2 electrons and 1 orbital (DQD21).
    """

    params: dict
    singleParticleLabels: list[str]
    singletTripletLabelsAndActivations: dict
    singletTripletLabelsOrdered: list[str]
    basesNames: list[str]

    def __init__(self, params: dict):
        self.MUB = 0.05788 # Bohr magneton in meV/Tesla
        self.nsToMeV = 1519.30 # hbar = 6.582x10-25 GeV s -> 1 GeV-1 = 6.582x10-15 s -> 1 ns = 1519.30 meV-1
        self.params = self._processParams(params)
        ManyBodyHamiltonian.__init__(self, norb=8, n_elec=2, hdict=self._build_single_particle_dict(), cdict=self._build_interaction_dict())
        DynamicsManager.__init__(self)
        self._initilizeGlobalLabels()
        self.basesNames = []
        self.basesSectors = []
        self.createBases()


    def _initilizeGlobalLabels(self):
        """
        Initializes the global labels for the single-particle states.
        The single-particle basis is (LUp+, LDown+, LUp-, LDown-, RUp+, RDown+, RUp-, RDown-)
        The singlet-triplet basis has been computed by hand and the corresponding orbitals and phases are stored in a dictionary.
        'LL,S,T0': {"orbitals": [[0,3], [1,2]], "phases": [1,- 1]} corresponds to the state c_0^† c_3^† - c_1^† c_2^† |0> = |LUp+, LDown-> - |LDown+, LUp->
        """

        self.singleParticleLabels = ['LUp+', 'LDown+', 'LUp-', 'LDown-', 'RUp+', 'RDown+', 'RUp-', 'RDown-']
        self.singletTripletLabelsAndActivations = {
            'LL,S,T0': {"orbitals": [[0,3], [1,2]], "phases": [1,- 1]},
            'LL,S,T+': {"orbitals": [[0,1]], "phases": [1]},
            'LL,S,T-': {"orbitals": [[2,3]], "phases": [1]},
            'LL,T0,S': {"orbitals": [[0,3], [1,2]], "phases": [1, 1]},
            'LL,T+,S': {"orbitals": [[0,2]], "phases": [1]},
            'LL,T-,S': {"orbitals": [[1,3]], "phases": [1]},
            'LR,S,T0': {"orbitals": [[0,7], [1,6], [2,5], [3,4]], "phases": [1, -1, 1, -1]},
            'LR,S,T+': {"orbitals": [[0,5], [1,4]], "phases": [1, -1]},
            'LR,S,T-': {"orbitals": [[2,7], [3,6]], "phases": [1, -1]},
            'LR,T0,S': {"orbitals": [[0,7], [1,6], [2,5], [3,4]], "phases": [1, 1, -1, -1]},
            'LR,T+,S': {"orbitals": [[0,6], [2,4]], "phases": [1, -1]},
            'LR,T-,S': {"orbitals": [[1,7], [3,5]], "phases": [1, -1]},
            'LR,S,S': {"orbitals": [[0,7], [1,6], [2,5], [3,4]], "phases": [1, -1, -1, 1]},
            'LR,T0,T0': {"orbitals": [[0,7], [1,6], [2,5], [3,4]], "phases": [1, 1, 1, 1]},
            'LR,T0,T+': {"orbitals": [[0,5], [1,4]], "phases": [1, 1]},
            'LR,T0,T-': {"orbitals": [[2,7], [3,6]], "phases": [1, 1]},
            'LR,T+,T0': {"orbitals": [[0,6], [2,4]], "phases": [1,1]},
            'LR,T+,T+': {"orbitals": [[0,4]], "phases": [1]},
            'LR,T+,T-': {"orbitals": [[2,6]], "phases": [1]},
            'LR,T-,T0': {"orbitals": [[3,5], [1,7]], "phases": [1, 1]},
            'LR,T-,T+': {"orbitals": [[1,5]], "phases": [1]},
            'LR,T-,T-': {"orbitals": [[3,7]], "phases": [1]},
            'RR,S,T0': {"orbitals": [[4,7], [5,6]], "phases": [1,- 1]},
            'RR,S,T+': {"orbitals": [[4,5]], "phases": [1]},
            'RR,S,T-': {"orbitals": [[6,7]], "phases": [1]},
            'RR,T0,S': {"orbitals": [[4,7], [5,6]], "phases": [1, 1]},
            'RR,T+,S': {"orbitals": [[4,6]], "phases": [1]},
            'RR,T-,S': {"orbitals": [[5,7]], "phases": [1]},
        }

    def _processParams(self, params: dict) -> dict:
        """
        Process and validate the input parameters dictionary.
        Ensures all required parameters are present and have valid types.
        If a required parameter is missing, uses the default value.
        If an unexpected parameter is present, raises an error.
        """
        defaults = {
            DQDParameters.B_Z.value: 0.0,
            DQDParameters.B_X.value: 0.0,
            DQDParameters.GS_L.value: 2.0,
            DQDParameters.GS_R.value: 2.0,
            DQDParameters.GV_L.value: 20.0,
            DQDParameters.GV_R.value: 20.0,
            DQDParameters.DELTA_SO.value: 0.066,
            DQDParameters.DELTA_KK.value: 0.02,
            DQDParameters.E_R.value: 0.0,
            DQDParameters.TUNNELING.value: 0.05,
            DQDParameters.T_SOC.value: 0.0,
            DQDParameters.U0.value: 6.0,
            DQDParameters.U1.value: 1.5,
            DQDParameters.X.value: 0.1,
            DQDParameters.A.value: 0.0,
            DQDParameters.P.value: 0.0,
            DQDParameters.J.value: 0.000075,
            DQDParameters.G_ORTHO.value: 10,
            DQDParameters.G_ZZ.value: 100,
            DQDParameters.G_Z0.value: 6.66,
            DQDParameters.G_0Z.value: 6.66
        }
        expected_params = set(defaults.keys())

        # Check for unexpected parameters
        for param in params:
            if param not in expected_params:
                raise ValueError(f"Unexpected parameter: {param}")

        # Fill missing parameters with defaults
        processed = {}
        for param in expected_params:
            if param in params:
                value = params[param]
            else:
                value = defaults[param]
            if not isinstance(value, (int, float)):
                raise TypeError(f"Parameter {param} must be a number (int or float)")
            processed[param] = value

        return processed
    
    def updateParams(self, new_params: dict):
        """
        Update the parameters of the DQD21 instance and rebuild the Hamiltonian.
        """
        all_params = deepcopy(self.params)
        for key, value in new_params.items():
            all_params[key] = value
        self.params = self._processParams(all_params)
        new_hdict = self._build_single_particle_dict()
        new_cdict = self._build_interaction_dict()
        self.update_hdict(new_hdict)
        self.update_cdict(new_cdict)

    def createBases(self):
        """
        Creates different bases for the many-body Hamiltonian.
        The Fock basis is already created when initializing the ManyBodyHamiltonian class.
        The singlet-triplet basis is created by hand and stored in a dictionary.
        We also create another order of the singlet-triplet basis where the states are sorted specifically for the singlet-triplet qubit.
        """

        # Add to the basis names the original basis
        self.basesNames.append(self.originalBasisName)
        self.basesSectors.append({'FockDeterminant': len(self.basesDict[self.originalBasisName]['labels'])})

        # Ordered singlet-triplet basis
        name = BasesNames.SINGLET_TRIPLET.name
        labels = ['LL,S,T0', 'LL,S,T+', 'LL,S,T-', 'LL,T0,S', 'LL,T+,S', 'LL,T-,S', # (2,0) configurations
            'LR,S,T0', 'LR,S,T+', 'LR,S,T-', 'LR,T0,S', 'LR,T+,S', 'LR,T-,S', # (1,1) configurations orbitally symmetric
            'LR,S,S', 'LR,T0,T0', 'LR,T0,T+', 'LR,T0,T-', 'LR,T+,T0', 'LR,T+,T+', 'LR,T+,T-', 'LR,T-,T0', 'LR,T-,T+', 'LR,T-,T-', # (1,1) configurations orbitally antisymmetric
            'RR,S,T0', 'RR,S,T+', 'RR,S,T-', 'RR,T0,S', 'RR,T+,S', 'RR,T-,S' # (0,2) configurations
            ]
        
        self.add_basis(basis_name=name, activations_dict=self.singletTripletLabelsAndActivations, labels=labels)
        self.basesNames.append(name)
        self.basesSectors.append({'(2,0)': 6, '(1,1)': 16, '(0,2)': 6})

        # Singlet-Triplet qubit basis is just a reordering of the previous basis with minimal 4 important states
        name = BasesNames.SINGLET_TRIPLET_QUBIT_4.name
        labels = [
            'LR,T-,T-', 'LR,S,T-', 'LL,S,T-', 'LR,T0,T-', # The working basis are the first 4 states
            'LL,S,T0', 'LR,S,T0', 'RR,S,T-', 'LR,T-,T0', 'LR,T+,T-', 'LR,T0,T0', # Direct interactions with minimal states
            'LR,T+,T0', 'LL,T-,S', 'LL,S,T+', 'LR,S,T+', 'LR,T0,T+', 'LR,T+,T+', 'LL,T0,S', 'LL,T+,S','LR,T0,S', 'LR,T-,S', 
            'LR,T+,S', 'LR,S,S', 'LR,T-,T+', 'RR,T-,S', 'RR,T0,S','RR,T+,S', 'RR,S,T0','RR,S,T+', # Rest of states
            ]
        
        self.add_basis(basis_name=name, activations_dict=self.singletTripletLabelsAndActivations, labels=labels)
        self.basesNames.append(name)
        self.basesSectors.append({'WorkingBasis': 4, 'DirectInteractions': 6, 'RestOfStates': 18})

        # Spatially symmetric and antisymmetric basis (reordering for coloring plots)

        name = BasesNames.SIMMETRIC_ANTISYMMETRIC.name
        labels = ['LL,S,T0', 'LL,S,T+', 'LL,S,T-', 'LL,T0,S', 'LL,T+,S', 'LL,T-,S', # (2,0) configurations
            'LR,S,T0', 'LR,S,T+', 'LR,S,T-', 'LR,T0,S', 'LR,T+,S', 'LR,T-,S', # (1,1) configurations orbitally symmetric
            'RR,S,T0', 'RR,S,T+', 'RR,S,T-', 'RR,T0,S', 'RR,T+,S', 'RR,T-,S', # (0,2) configurations
            'LR,S,S', 'LR,T0,T0', 'LR,T0,T+', 'LR,T0,T-', 'LR,T+,T0', 'LR,T+,T+', 'LR,T+,T-', 'LR,T-,T0', 'LR,T-,T+', 'LR,T-,T-', # (1,1) configurations orbitally antisymmetric
            ]
        
        self.add_basis(basis_name=name, activations_dict=self.singletTripletLabelsAndActivations, labels=labels)
        self.basesNames.append(name)
        self.basesSectors.append({'Symmetric': 18, 'Antisymmetric': 10})


        # Singlet-Triplet qubit basis is just a reordering of the s-t basis but conserving all the valley T- (2,0)-(1,1) states for completitude
        name = BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name
        labels = [
            'LR,T-,T-', 'LR,S,T-', 'LL,S,T-', 'LR,T0,T-', 'LR,T+,T-', # The working basis are the first 5 states
            'LL,S,T0', 'LR,S,T0', 'RR,S,T-', 'LR,T-,T0', 'LR,T0,T0', 'LR,T+,T0', # Direct interactions with minimal states
            'LL,T-,S', 'LL,S,T+', 'LR,S,T+', 'LR,T0,T+', 'LR,T+,T+', 'LL,T0,S', 'LL,T+,S','LR,T0,S', 'LR,T-,S', 
            'LR,T+,S', 'LR,S,S', 'LR,T-,T+', 'RR,T-,S', 'RR,T0,S','RR,T+,S', 'RR,S,T0','RR,S,T+', # Rest of states
            ]
        
        self.add_basis(basis_name=name, activations_dict=self.singletTripletLabelsAndActivations, labels=labels)
        self.basesNames.append(name)
        self.basesSectors.append({'WorkingBasis': 5, 'DirectInteractions': 6, 'RestOfStates': 16})


    def get_current_indices(self, basis_name: str, cutOffN: int) -> Tuple[List[int], List[int]]: 
        listOfSymmetricIndices = []
        listOfAntiSymmetricIndices = []

        listOfStatesInParticularBasis = self.basesDict[basis_name]["labels"][:cutOffN]
        listOfSymmetricStates = self.basesDict[BasesNames.SIMMETRIC_ANTISYMMETRIC.name]["labels"][:18]
        listOfAntiSymmetricStates = self.basesDict[BasesNames.SIMMETRIC_ANTISYMMETRIC.name]["labels"][18:]

        for idx, state in enumerate(listOfStatesInParticularBasis):
            if state in listOfSymmetricStates:
                listOfSymmetricIndices.append(idx)
            elif state in listOfAntiSymmetricStates:
                listOfAntiSymmetricIndices.append(idx)
            else:
                raise ValueError(f"State {state} not recognized in singlet-triplet symmetry basis.")
            
        return listOfSymmetricIndices, listOfAntiSymmetricIndices


    def _build_single_particle_dict(self):
        """
        Constructs the single-particle Hamiltonian dictionary for the double quantum dot system.
        The basis is (LUp+, LDown+, LUp-, LDown-, RUp+, RDown+, RUp-, RDown-)
        """
        b_field = self.params[DQDParameters.B_Z.value]
        b_parallel = self.params[DQDParameters.B_X.value]
        gsL = self.params[DQDParameters.GS_L.value]
        gsR = self.params[DQDParameters.GS_R.value]
        gvL = self.params[DQDParameters.GV_L.value]
        gvR = self.params[DQDParameters.GV_R.value]
        DeltaSO = self.params[DQDParameters.DELTA_SO.value]
        DeltaKK = self.params[DQDParameters.DELTA_KK.value]
        t = self.params[DQDParameters.TUNNELING.value]
        t_soc = self.params[DQDParameters.T_SOC.value]
        ER = self.params[DQDParameters.E_R.value]

        spin_zeeman_splitting_left = 0.5 * gsL * self.MUB * b_field
        spin_parallel_splitting_left = 0.5 * gsL * self.MUB * b_parallel
        valley_zeeman_splitting_left = 0.5 * gvL * self.MUB * b_field
        spin_zeeman_splitting_right = 0.5 * gsR * self.MUB * b_field
        spin_parallel_splitting_right = 0.5 * gsR * self.MUB * b_parallel
        valley_zeeman_splitting_right = 0.5 * gvR * self.MUB * b_field
        kane_mele_splitting = 0.5 * DeltaSO

        # Intradot dynamics
        h = {(0, 0): kane_mele_splitting + valley_zeeman_splitting_left + spin_zeeman_splitting_left,  # LUp+
         (1, 1): - kane_mele_splitting + valley_zeeman_splitting_left - spin_zeeman_splitting_left,  # LDown+
         (2, 2): - kane_mele_splitting - valley_zeeman_splitting_left + spin_zeeman_splitting_left,  # LUp-
         (3, 3): kane_mele_splitting - valley_zeeman_splitting_left - spin_zeeman_splitting_left,  # LDown-
         (4, 4): ER + kane_mele_splitting + valley_zeeman_splitting_right + spin_zeeman_splitting_right, # RUp+
         (5, 5): ER - kane_mele_splitting + valley_zeeman_splitting_right - spin_zeeman_splitting_right, # RDown+
         (6, 6): ER - kane_mele_splitting - valley_zeeman_splitting_right + spin_zeeman_splitting_right, # RUp-
         (7, 7): ER + kane_mele_splitting - valley_zeeman_splitting_right - spin_zeeman_splitting_right, # RDown-
         (0, 1): spin_parallel_splitting_left,
         (2, 3): spin_parallel_splitting_left,
         (4, 5): spin_parallel_splitting_right,
         (6, 7): spin_parallel_splitting_right,
         (0, 2): DeltaKK,
         (1, 3): DeltaKK,
         (4, 6): DeltaKK,
         (5, 7): DeltaKK,
         }

        # Interdot dynamics
        h.update({
            (0, 4): t,  # LUp+ ↔ RUp+
            (1, 5): t,  # LDown+ ↔ RDown+
            (2, 6): t,  # LUp- ↔ RUp-
            (3, 7): t   # LDown- ↔ RDown-
        })

        h.update({
            (0, 5): 1j * t_soc,   # LUp+ → RDown+
            (1, 4): -1j * t_soc,   # LDown+ → RUp+
            (2, 7): 1j * t_soc,    # LUp- → RDown-
            (3, 6): -1j * t_soc,   # LDown- → RUp-
        })


        return h
    
    def _build_interaction_dict(self):
        """
        Constructs the interaction Hamiltonian dictionary for the double quantum dot system.
        Is uses the ordering in the basis given by the many body Hamiltonian.

        The terms (h,j,k,m) only fulfills j>h as some processes (direct vs exchange) make the difference between k>m and m>k 
        Some possible ill-defined terms are commented below
        """

        U0 = self.params[DQDParameters.U0.value]
        U1 = self.params[DQDParameters.U1.value]
        X = self.params[DQDParameters.X.value]
        A = self.params[DQDParameters.A.value]
        P = self.params[DQDParameters.P.value]
        J = self.params[DQDParameters.J.value]
        g_ortho = self.params[DQDParameters.G_ORTHO.value]
        g_zz = self.params[DQDParameters.G_ZZ.value]
        g_z0 = self.params[DQDParameters.G_Z0.value]
        g_0z = self.params[DQDParameters.G_0Z.value]

        V = {}

        # Intra dot coulomb terms
        coulomb_intradot_intravalley = [
            (0, 1, 1, 0), (2, 3, 3, 2),
            (4, 5, 5, 4), (6, 7, 7, 6),
        ]

        for (h,j,k,m) in coulomb_intradot_intravalley:
            V[(h,j,k,m)] =  U0 + J*(g_zz + g_z0 + g_0z)

        coulomb_intradot_intervalley = [
            (0, 2, 2, 0), (0, 3, 3, 0), (1, 2, 2, 1), (1, 3, 3, 1),
            (4, 6, 6, 4), (4, 7, 7, 4), (5, 6, 6, 5), (5, 7, 7, 5)
        ]

        for (h,j,k,m) in coulomb_intradot_intervalley:
            V[(h,j,k,m)] =  U0 + J*(g_zz - g_z0 - g_0z)

        ortho_exchanges = [
            (0, 2, 0, 2), (0, 3, 1, 2), (1, 3, 1, 3), (1, 2, 0, 3), # (0,3,1,2) and (1,2,0,3) correspond to the same entry as they are complex conjugates
            # Should they be (0,3,1,2) and (0,3,2,1)? With the same sign?
            (4, 6, 4, 6), (4, 7, 5, 6), (5, 7, 5, 7), (5, 6, 4, 7), # Same for right dot as it is a copy of the left one
        ]

        for (h,j,k,m) in ortho_exchanges:
            V[(h,j,k,m)] =  4*J*g_ortho

        # Interdot coulomb terms
        interdot_coulomb = [
            (0, 4, 4, 0), (0, 5, 5, 0), (0, 6, 6, 0), (0, 7, 7, 0),
            (1, 4, 4, 1), (1, 5, 5, 1), (1, 6, 6, 1), (1, 7, 7, 1),
            (2, 4, 4, 2), (2, 5, 5, 2), (2, 6, 6, 2), (2, 7, 7, 2),
            (3, 4, 4, 3), (3, 5, 5, 3), (3, 6, 6, 3), (3, 7, 7, 3)
        ]
        for (h,j,k,m) in interdot_coulomb:
            V[(h,j,k,m)] =  U1

        # Exhange terms
        exchange = [
            (0, 4, 0, 4), (1, 5, 1, 5), (2, 6, 2, 6), (3, 7, 3, 7)
        ]
        for (h,j,k,m) in exchange:
            V[(h,j,k,m)] =  X

        # Assisted hopping terms
        # All the terms respect the convention h<j
        # Added term for an electron staying in the right dot maintaining the convention
        # As assisted hopping is a critical parameter for the anticrossing, we have to be sure of its terms
        # For example: does it make sense that the amplitude of the term (A) is the same if: 
        # 1) the two electrons start in the same dot and one of them tunnel to the other
        # 2) each electron is in one dot and one of them tunnel to the other, occupying the same dot

        densityAssistedEntries = [
            (0, 1, 5, 0), (0, 2, 6, 0), (0, 3, 7, 0), (0, 5, 1, 0), (0, 6, 2, 0), (0, 7, 3, 0),
            (1, 2, 6, 1), (1, 3, 7, 1), (1, 4, 0, 1), (1, 6, 2, 1), (1, 7, 3, 1), 
            (2, 3, 7, 2), (2, 4, 0, 2), (2, 5, 1, 2), (2, 7, 3, 2),
            (3, 4, 0, 3), (3, 5, 1, 3), (3, 6, 2, 3),
            (4, 5, 1, 4), (4, 6, 2, 4), (4, 7, 3, 4),
            (5, 6, 2, 5), (5, 7, 3, 5),
            (6, 7, 3, 6)
        ]

        for (h,j,k,m) in densityAssistedEntries:
            V[(h,j,k,m)] =  A

        # Pair hopping terms
        # As k>m (in h,j,k,m) is not required anymore, should terms like (0, 1, 4, 5) be included?

        pair_hopping = [
            (0, 1, 5, 4), (0, 2, 6, 4), (0, 3, 7, 4),
            (1, 2, 6, 5), (1, 3, 7, 5), (2, 3, 7, 6)
        ]
        for (h,j,k,m) in pair_hopping:
            V[(h,j,k,m)] =  P

        return V

    def calculate_eigenvalues_and_eigenvectors(self, parameterToChange: str = None, newValue: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the eigenvalues and eigenvectors of the Hamiltonian.
        If parameterToChange and newValue are provided, update the parameter before calculation.
        """

        if parameterToChange is not None:
            if newValue is None:
                raise ValueError("Parameter to change introduced with no new value.")
            self.updateParams({parameterToChange: newValue})
        return ManyBodyHamiltonian.calculate_eigenvalues_and_eigenvectors(self)


    def getHamiltonianInBase(self, basisName: str) -> np.ndarray:
        """
        Returns the Hamiltonian matrix in the specified basis.
        """
        if basisName not in self.basisNames:
            raise ValueError(f"Basis {basisName} not found. Available bases: {self.basesNames}")
        return self.project_operator_to_basis(self.matrixHamiltonian, basisName)
    
    def schriefferWolffTransformation(self, listOfOperators = None, basis: Optional[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Perform the SWT to obtain the effective Hammiltonian (and possibly other operators) 
        in our selected (4x4) basis for the singlet-triplet qubit.
        """

        if basis is not None:
            if basis == BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name:
                H_full = self.getHamiltonianInBase(BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name)
                return ManyBodyHamiltonian.schriefferWolffTransformation(self, H_full, N0=5, N1 = 6, listOfExtraOperators=listOfOperators)

        H_full = self.getHamiltonianInBase(BasesNames.SINGLET_TRIPLET_QUBIT_4.name)

        return ManyBodyHamiltonian.schriefferWolffTransformation(self, H_full, N0=4, N1 = 6, listOfExtraOperators=listOfOperators)

    def __repr__(self):
        return (f"BLG Double Quantum Dot with 2 electrons and 1 orbital per dot (DQD21)\n"
                f"Single-particle basis: {self.singleParticleLabels}\n"
                f"Parameters (meV/T): {self.params}\n")
    

    def performTimeEvolution(self, options: dict, basis: Optional[str] = None)-> Result:
        """
        Perform a time evolution of the system using QuTiP.
        It receives a dictionary with the options for the time evolution.
        The dictionary must contain
        - 'cutOffN' (int): Cut-off for the number of states to consider in the time evolution.
          If no specified or None, use the SWT effective Hamiltonian.
        - 'parameterToChange' (str): Parameter to change during the time evolution.
          If None, no parameter is changed.
        - 'protocolDetails' (list): Details of the protocol to change the parameter. 
            Each element is a list of floats of the form [initialValue, finalValue, duration].
            If parameterToChange is None, only the duration is considered.
        - 'initialState' (dict): Specific parameters to define the initial state.
          If empty or None, the ground state of the initial Hamiltonian is used.
        - 'totalPoints' (int): Total number of points in the time evolution.
        - 'dephasingRate' (float): Global dephasing rate to consider in the time evolution.
          If empty or None, no dephasing is considered.
        - 'spinRelaxationRate' (float): Global spin relaxation rate to consider in the time evolution.
          If empty or None, no spin relaxation is considered.
        """

        cutOffN = options.get('cutOffN', None)
        parameterToChange = options.get('parameterToChange', None)
        protocolDetails = options.get('protocolDetails', [[0.0, 0.0, 1.0]])
        initialStateParameters = options.get('initialState', {})
        totalPoints = options.get('totalPoints', 100)
        dephasingRate = options.get('dephasingRate', None)
        spinRelaxationRate = options.get('spinRelaxationRate', None)

        # --- temporal protocol ---
        if parameterToChange is None:
            tlistNano = np.linspace(0, protocolDetails[0][2], totalPoints)
            tlist = tlistNano * self.nsToMeV
            # Fixed Hamiltonian case
            hEff = self._get_effective_hamiltonian(cutOffN, basis=basis)
            hEffQobj = Qobj(hEff)
            rho0 = self._get_initial_state(hEffQobj, initialStateParameters, cutOffN, basis=basis)
            collapseOperators = self._getCollapseOperators(cutOffN, dephasingRate, spinRelaxationRate, basis=basis)
            return DynamicsManager.performTimeEvolution(self, initial_state=rho0, times=tlist, unique_hamiltonian=hEffQobj, collapse_operators=collapseOperators)
        else:
            tlistNano, parameterValues = DynamicsManager.build_protocol_timeline(protocolDetails, totalPoints)
            tlist = tlistNano * self.nsToMeV
            listOfHamiltonians = [Qobj(self._get_effective_hamiltonian(cutOffN, parameterToChange, value, basis=basis)) for value in parameterValues]
            rho0 = self._get_initial_state(listOfHamiltonians[0], initialStateParameters, cutOffN, basis=basis)
            collapseOperators = self._getCollapseOperators(cutOffN, dephasingRate, spinRelaxationRate, basis=basis)
            return DynamicsManager.performTimeEvolution(self, initial_state=rho0, times=tlist, precomputed_hamiltonians=listOfHamiltonians, collapse_operators=collapseOperators)

    def _get_effective_hamiltonian(self, cutOffN, parameterToChange=None, value=None, basis: Optional[str] = None):
        """
        Returns the effective Hamiltonian in the singlet-triplet qubit basis.
        If cutOffN is provided, returns the Hamiltonian projected to the first cutOffN states.
        If parameterToChange and value are provided, updates the parameter before calculating the Hamiltonian.
        """
        originalParams = deepcopy(self.params)
        if parameterToChange is not None and value is not None:
            self.updateParams({parameterToChange: value})
        if cutOffN is not None:
            if basis is not None:
                if basis == BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name:
                    H_full = self.getHamiltonianInBase(BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name)
            else:
                H_full = self.getHamiltonianInBase(BasesNames.SINGLET_TRIPLET_QUBIT_4.name)
            self.updateParams(originalParams)  # Restore original parameters
            return H_full[:cutOffN, :cutOffN]
        else:
            hEff, _ = self.schriefferWolffTransformation(basis=basis)
            self.updateParams(originalParams)  # Restore original parameters
            return hEff

    # TODO: Averiguar por qué el estado inicial no se calcula bien. Una vez hecho eso, se puede pasar al analisis de frecuencias

    def _get_initial_state(self, hamiltonianQobj, initialStateParameters, cutOffN, basis: Optional[str] = None):
        """
        Returns the initial state for the time evolution.
        If initialStateParameters is provided, updates the parameters before calculating the ground state.
        If initialStateParameters is empty or None, returns the ground state of the provided Hamiltonian.
        """
        originalParams = deepcopy(self.params)
        if initialStateParameters:
            self.updateParams(initialStateParameters)
            if cutOffN is not None:
                if basis is not None:
                    if basis == BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name:
                        H_full = self.getHamiltonianInBase(BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name)
                else:
                    H_full = self.getHamiltonianInBase(BasesNames.SINGLET_TRIPLET_QUBIT_4.name)
                hEffModified = H_full[:cutOffN, :cutOffN]
            else:
                hEffModified, _ = self.schriefferWolffTransformation(basis=basis)
            hEffQobjInitial = Qobj(hEffModified)
        else:
            hEffQobjInitial = hamiltonianQobj
        _, evecs = hEffQobjInitial.eigenstates()
        psi0 = evecs[0]

        self.updateParams(originalParams)  # Restore original parameters
        return psi0 * psi0.dag()



    def _getCollapseOperators(self, cutOffN, dephasingRate = None, spinRelaxationRate = None, basis: Optional[str]=None) -> List[Qobj]:
        """
        Returns the list of collapse operators for the time evolution.
        If cutOffN is provided, returns the collapse operators projected to the first cutOffN states.
        If dephasingRate or spinRelaxationRate are provided, includes the corresponding collapse operators.
        """
        listOfOperators = []
        if dephasingRate is not None and dephasingRate > 0:
             # Dephasing operators in the full basis
            for i in range(8):
                Li = self.buildDecoherenceOperator(i, i)
                if basis is not None:
                    if basis == BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name:
                        Li_proj = np.sqrt(dephasingRate) * self.project_operator_to_basis(BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name, Li)
                else:
                    Li_proj = np.sqrt(dephasingRate) * self.project_operator_to_basis(BasesNames.SINGLET_TRIPLET_QUBIT_4.name, Li)
                listOfOperators.append(Li_proj)
    
        if spinRelaxationRate is not None and spinRelaxationRate > 0:
            # Spin relaxation operators in the full basis
            spin_relaxation_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
            for (down, up) in spin_relaxation_pairs:
                Li = self.buildDecoherenceOperator(up, down)
                if basis is not None:
                    if basis == BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name:
                        Li_proj = np.sqrt(spinRelaxationRate) * self.project_operator_to_basis(BasesNames.SINGLET_TRIPLET_QUBIT_VALLEY_MINUS.name, Li)
                else:
                    Li_proj = np.sqrt(spinRelaxationRate) * self.project_operator_to_basis(BasesNames.SINGLET_TRIPLET_QUBIT_4.name, Li)
                
                listOfOperators.append(Li_proj)

        if cutOffN is not None:
            listOfOperators = [L[:cutOffN, :cutOffN] for L in listOfOperators]
            # Remove zero operators
            listOfOperators = [L for L in listOfOperators if np.linalg.norm(L) > 1e-10]

        else:
            listOfOperators = self.schriefferWolffTransformation(listOfOperators=listOfOperators, basis=basis)[1]

        return [Qobj(L) for L in listOfOperators]
    
    def gamma_from_time(self, t_ns: float) -> float:
        return DynamicsManager.gamma_from_time(t_ns, self.nsToMeV)
    