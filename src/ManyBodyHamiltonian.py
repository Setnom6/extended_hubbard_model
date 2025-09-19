from __future__ import annotations

import itertools
from typing import Dict, List, Tuple
from pymablock import block_diagonalize, operator_to_BlockSeries, series

from scipy.linalg import eigh
import numpy as np

OneBodyDict = Dict[Tuple[int, int], complex]
TwoBodyDict = Dict[Tuple[int, int, int, int], complex]
BasesDict = Dict[str, Dict[str, List[str] | List[np.ndarray]]]

from .FockSpaceUtilities import FockSpaceUtilities

class ManyBodyHamiltonian:

    norb: int # number of orbitals
    n_elec: int # number of electrons
    hdict: OneBodyDict # one-body Hamiltonian dictionary
    cdict: TwoBodyDict # two-body Coulomb interaction dictionary
    basesDict: BasesDict # dictionary of possible bases
    matrixHamiltonian: np.ndarray | None # resulting built Hamiltonian matrix
    originalBasisName: str # name of the original basis (determinant basis)

    """
    Represents a many-body Hamiltonian operator.

    This class is designed to construct and manipulate the Hamiltonian of a quantum
    many-body system, using one-body and two-body interaction terms. The Hamiltonian
    matrix can be generated based on a specific basis of determinant. It provides tools to construct the
    system's basis and build the resulting sparse Hamiltonian matrix.
    """

    def __init__(self, norb: int, n_elec: int,  hdict: OneBodyDict, cdict: TwoBodyDict):

        self.originalBasisName = 'FOCK_DETERMINANT'
        self.norb = norb
        self.n_elec = n_elec
        self.hdict = hdict
        self.cdict = cdict
        self.basesDict = self.initialize_bases_dict()
        self.matrixHamiltonian = self.buildMatrixHamiltonian(self.hdict, self.cdict)

    def update_hdict_entry(self, key: Tuple[int, int], value: complex):
        """Update an element of the hdict and rebuild the matrix."""
        self.hdict[key] = value
        self.matrixHamiltonian = self.buildMatrixHamiltonian(self.hdict, self.cdict)

    def update_cdict_entry(self, key: Tuple[int, int, int, int], value: complex):
        """Update an element of the cdict and rebuild the matrix."""
        self.cdict[key] = value
        self.matrixHamiltonian = self.buildMatrixHamiltonian(self.hdict, self.cdict)

    def update_hdict(self, new_hdict: OneBodyDict):
        """Update the entire hdict and rebuild the matrix."""
        self.hdict = new_hdict
        self.matrixHamiltonian = self.buildMatrixHamiltonian(self.hdict, self.cdict)

    def update_cdict(self, new_cdict: TwoBodyDict):
        """Update the entire cdict and rebuild the matrix."""
        self.cdict = new_cdict
        self.matrixHamiltonian = self.buildMatrixHamiltonian(self.hdict, self.cdict)

    def initialize_bases_dict(self) -> BasesDict:
        """Initialize dictionary of possible bases.
        
        :return: A dictionary containing different basis representations.
        :rtype: BasesDict
        """
        FSUtils = FockSpaceUtilities(self.norb, self.n_elec)
        basis = FSUtils.basis
        dim = len(basis)

        # Correspondence between determinant basis and labels
        correspondence_with_determinant_basis = {
            i: ''.join(str((basis[i] >> j) & 1) for j in range(self.norb))
            for i in range(dim)
        }

        # Correspondence between determinant basis and vectors
        determinant_basis_as_vectors = [
            np.array([1 if j == i else 0 for j in range(dim)], dtype=np.complex128)
            for i in range(dim)
        ]

        return {
            self.originalBasisName: {
                'labels': list(correspondence_with_determinant_basis.values()),
                'vectors': determinant_basis_as_vectors
            }
        }

    def buildMatrixHamiltonian(self, hdict: OneBodyDict, cdict: TwoBodyDict) -> np.ndarray:
        FSUtils = FockSpaceUtilities(self.norb, self.n_elec)

        h1 = self.complete_hermitian_1body(hdict, self.norb)
        V = self.coulomb_tensor_from_dict(cdict, self.norb)
        
        basis = FSUtils.basis
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=np.complex128)
        idx = {det: i for i, det in enumerate(basis)}

        # one‑body part
        for p in range(self.norb):
            for q in range(self.norb):
                hval = h1[p, q]
                if hval == 0:
                    continue
                for j, det in enumerate(basis):
                    t1, s1 = FSUtils.apply_annihilation(det, q)
                    if t1 is None:
                        continue
                    t2, s2 = FSUtils.apply_creation(t1, p)
                    if t2 is None:
                        continue
                    H[idx[t2], j] += hval * s1 * s2

        # two‑body part
        for p, r, q, s in itertools.product(range(self.norb), repeat=4):
            v = V[p, r, q, s]
            if v == 0:
                continue
            for j, det in enumerate(basis):
                t1, s1 = FSUtils.apply_annihilation(det, s)
                if t1 is None:
                    continue
                t2, s2 = FSUtils.apply_annihilation(t1, q)
                if t2 is None:
                    continue
                t3, s3 = FSUtils.apply_creation(t2, r)
                if t3 is None:
                    continue
                t4, s4 = FSUtils.apply_creation(t3, p)
                if t4 is None:
                    continue
                H[idx[t4], j] += 0.5 * v * s1 * s2 * s3 * s4

        matrix = (H + H.T.conj()) / 2.0
        return matrix
    
    @staticmethod
    def complete_hermitian_1body(hdict: OneBodyDict, norb: int) -> np.ndarray:
        """
        Complete a Hermitian one-body operator matrix given its upper triangular part.

        This function takes a dictionary representing the upper triangular part
        of a Hermitian matrix in one-body operator notation and reconstructs the
        complete Hermitian matrix as a NumPy array. In Hermitian matrices, the
        element at position `(p, q)` must equal the complex conjugate of the element
        at position `(q, p)`. The function ensures that this property is preserved
        while completing the full matrix from the given input data.

        The input dictionary `hdict` should only contain keys where `p ≤ q`, as it
        represents the upper triangular part. If any key is found with `p > q`,
        a `ValueError` is raised. This ensures the validity of the input format.

        :param hdict: A dictionary where keys are tuples `(p, q)` with `p ≤ q`
                    and values are the corresponding matrix elements. Represents
                    the upper triangular part of a Hermitian matrix.
        :type hdict: OneBodyDict
        :param norb: The total number of orbitals, defining the dimensions of the
                    resulting matrix as `norb x norb`.
        :type norb: int
        :return: A complex-valued Hermitian matrix of shape `(norb, norb)` constructed
                from the input dictionary.
        :rtype: numpy.ndarray
        :raises ValueError: If a key `(p, q)` in the input dictionary satisfies `p > q`.
        """
        H = np.zeros((norb, norb), dtype=np.complex128)
        for (p, q), val in hdict.items():
            if p > q:
                raise ValueError("One‑body dict must only contain keys with p ≤ q")
            H[p, q] = val
            if p != q:
                H[q, p] = np.conjugate(val)
        return H
    
    @staticmethod
    def coulomb_tensor_from_dict(cdict: TwoBodyDict, norb: int) -> np.ndarray:
        """
        Generates the Coulomb tensor in the two-electron integral format from a given
        two-body dictionary representation. The resulting tensor includes both the
        provided integrals and their Hermitian counterparts by ensuring Hermitian
        symmetry. 

        :param cdict: Dictionary representation of two-body integrals. The keys are
            tuples of orbital indices (p, r, q, s), and the values are the corresponding
            complex integral values. The dict have to fulfill the convention r>p, q>s
            to avoid over sumation in the production of the Hamiltonian afterwards.
        :type cdict: TwoBodyDict
        :param norb: Total number of orbitals. Determines the size of the resulting
            Coulomb tensor.
        :type norb: int
        :return: A 4-dimensional NumPy array of shape (norb, norb, norb, norb),
            representing the Coulomb tensor. The tensor is Hermitian with respect to
            its bra and ket indices.
        :rtype: numpy.ndarray
        """
        V = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
        for (p, r, q, s), val in cdict.items():
            # Hermitian counterpart (bra ↔ ket)
            V[p, r, q, s] = val
            V[q, s, p, r] = np.conjugate(val)
        return V

    def add_basis(self, basis_name: str, labels: List[str], activations_dict: Dict[str, Dict[str, List]]) -> None:
        """
        Given a basis name, an ORDERED list of labels and a dictionary which matches each label 
        with a list of activations in the determinant basis, this function adds the new basis to the
        basesDict attribute of the ManyBodyHamiltonian instance.

        :param basis_name: Name of the new basis to be added.
        :type basis_name: str
        :param labels: Ordered list of labels for the new basis.
        :type labels: List[str]
        :param activations_dict: Dictionary mapping each label to a dictionary with keys 'orbitals' and 'phases'.
            The 'orbitals' key should map to a list of tuples of orbitals to be activated,
            and the 'phases' key should map to a list of corresponding coefficients for these activations.
        :type activations_dict: Dict[str, Dict[str, List]]
        """

        if basis_name in self.basesDict:
            raise ValueError(f"Basis '{basis_name}' already exists in basesDict.")
        
        dim = len(self.basesDict[self.originalBasisName]['labels'])
        new_basis_vectors = np.zeros((dim, len(labels)), dtype=np.complex128)
        
        FSUtils = FockSpaceUtilities(self.norb, self.n_elec)
        determinant_basis = FSUtils.basis
        det_index = {det: i for i, det in enumerate(determinant_basis)}
        
        for col, label in enumerate(labels):
            if label not in activations_dict:
                raise ValueError(f"Label '{label}' not found in activations_dict.")
            
            orbitals_list = activations_dict[label].get('orbitals', [])
            phases_list = activations_dict[label].get('phases', [])
            
            if len(orbitals_list) != len(phases_list):
                raise ValueError(f"Mismatch in lengths of 'orbitals' and 'phases' for label '{label}'.")
            
            for orbitals, phase in zip(orbitals_list, phases_list):
                state = FSUtils.create_state_from_occupied_orbitals(orbitals)
                if state not in det_index:
                    raise ValueError(f"State with orbitals {orbitals} is not in the determinant basis.")
                
                row = det_index[state]
                new_basis_vectors[row, col] += phase
        
        # Normalize each vector
        for col in range(new_basis_vectors.shape[1]):
            norm = np.linalg.norm(new_basis_vectors[:, col])
            if norm > 0:
                new_basis_vectors[:, col] /= norm

        # Check orthonormality with detailed diagnostics
        overlap_matrix = new_basis_vectors.conj().T @ new_basis_vectors
        identity = np.eye(overlap_matrix.shape[0])
        atol = 1e-8
        if not np.allclose(overlap_matrix, identity, atol=atol):
            msg = [f"Orthonormality error in basis '{basis_name}':"]
            # Check for non-unit norm
            for i, label in enumerate(labels):
                norm = np.linalg.norm(new_basis_vectors[:, i])
                if not np.isclose(norm, 1.0, atol=atol):
                    msg.append(f"  Vector '{label}' (index {i}) has norm {norm:.6g} (should be 1)")
            # Check for non-orthogonal pairs
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    overlap = overlap_matrix[i, j]
                    if not np.isclose(overlap, 0.0, atol=atol):
                        msg.append(f"  Vectors '{labels[i]}' (index {i}) and '{labels[j]}' (index {j}) are not orthogonal: overlap = {overlap:.6g}")
            raise ValueError('\n'.join(msg))
        
        self.basesDict[basis_name] = {
            'labels': labels,
            'vectors': [new_basis_vectors[:, i] for i in range(new_basis_vectors.shape[1])]
        }

    @property
    def basisNames(self) -> List[str]:
        """Returns a list of the names of the available bases."""
        return list(self.basesDict.keys())

    def project_operator_to_basis(self, operator: np.ndarray, basis_name: str):
        Umatrix = np.array(self.basesDict[basis_name]['vectors']).T
        assert np.allclose(Umatrix.conj().T @ Umatrix, np.eye(Umatrix.shape[1])), "The transformation matrix is not unitary. Probably the basis is not orthonormal."

        return Umatrix.conj().T @ operator @ Umatrix
    
    def buildDecoherenceOperator(self, fromOrbital: int, toOrbital: int) -> np.ndarray:
        """
        Build the decoherence Lindblad operator: L = c_to† c_from
        The intensity of the decoherence should be applied externally.

        Args:
            fromOrbital: orbital to annihilate (j)
            toOrbital: orbital to create (i)

        Returns:
            np.ndarray representing the Lindblad operator in Fock basis
        """
        size = len(self.basis)
        op = np.zeros((size, size), dtype=complex)

        for i, state in enumerate(self.basis):
            newState, phase1 = self.fockUtils.apply_annihilation(state, fromOrbital)
            if newState is None:
                continue

            finalState, phase2 = self.fockUtils.apply_creation(newState, toOrbital)
            if finalState is None:
                continue

            j = self.indexMap.get(finalState)
            if j is not None:
                op[j, i] = phase1 * phase2

        return op
    
    def schriefferWolffTransformation(self, H_full: np.ndarray, N0: int, N1: int, 
                                      listOfExtraOperators: List[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Using Pymablock, perform a Schrieffer-Wolff transformation to block-diagonalize the Hamiltonian.
        The first step disregard completely the last block (any state with index higher than N0+N1).
        The second step block-diagonalize the first two blocks (N0 and N1).
        Finally, it returns the effective (N0 x N0) Hamiltonian.

        If a list of extra operators is provided, it also transforms them to the effective basis.
        """

        if listOfExtraOperators is None:
            listOfExtraOperators = []

        subspace_indices = [0]*N0 + [1]*N1
        H0_tot = H_full[:N0+N1, :N0+N1]
        H0 = np.diag(np.diag(H0_tot))
        H1 = H0_tot-H0

        hamiltonian = [H0, H1]
        H_tilde, U, U_adj = block_diagonalize(hamiltonian, subspace_indices=subspace_indices)
        try:
            transformed_H = np.ma.sum(H_tilde[:2, :2, :3], axis=2)
        except:
            transformed_H = np.ma.sum(H_tilde[:2, :2, :2], axis=2)

        listToReturn = []
        for operator in listOfExtraOperators:
            operator_reduced = operator[:N0+N1, :N0+N1]
            if np.allclose(np.abs(operator_reduced), 0, atol=1e-10):
                    continue # Skip zero operators which are outside the subspace
            op_series = operator_to_BlockSeries([np.diag(np.diag(operator_reduced)), 
                                                operator_reduced-np.diag(np.diag(operator_reduced))], 
                                                hermitian=True, subspace_indices=subspace_indices)
            operator_tilde = series.cauchy_dot_product(U_adj, op_series, U)
            try: 
                operator_eff = np.ma.sum(operator_tilde[:2, :2, :3], axis=2)
            except:
                operator_eff = np.ma.sum(operator_tilde[:2, :2, :2], axis=2)

            if np.ma.is_masked(operator_eff) or np.all(operator_eff.mask):
                    continue # Skip masked or zero operators
            listToReturn.append(operator_eff[0,0]) # Project to the effective (N0 x N0) space

        return transformed_H[0,0], listToReturn

    def calculate_eigenvalues_and_eigenvectors(self) -> Tuple[np.ndarray, np.ndarray]:
        eigval, eigv = eigh(self.matrixHamiltonian)
        return eigval, eigv
    

    def classify_eigenstate_in_basis(self, eigenstate: np.ndarray, basis_name: str) -> List[dict]:
        """
        Classifies an eigenvector (expressed in the Fock/determinant basis) in terms of an alternative basis.

        Args:
            eigenstate: np.ndarray, vector of N components in the Fock basis.
            basis_name: str, name of the alternative basis (must be in self.basisNames).

        Returns:
            List[dict]: Sorted list of dictionaries with 'label' and 'probability' for each vector in the selected basis.
        """
        if basis_name not in self.basisNames:
            raise ValueError(f"Basis '{basis_name}' not found. Available bases: {self.basisNames}")

        basis_vectors = self.basesDict[basis_name]['vectors']
        basis_labels = self.basesDict[basis_name]['labels']

        probabilities = []
        for label, vector in zip(basis_labels, basis_vectors):
            # Compute overlap between the eigenstate and the basis vector
            overlap = np.vdot(vector, eigenstate)
            probability = np.abs(overlap) ** 2
            probabilities.append({'label': label, 'probability': probability})

        # Sort by probability in descending order
        probabilities_sorted = sorted(probabilities, key=lambda x: x['probability'], reverse=True)
        return probabilities_sorted
