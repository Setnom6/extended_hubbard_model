from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple



BitDet = int

class FockSpaceUtilities():
    norb: int # number of orbitals (or single-particle states)
    nelec: Optional[int | Sequence[int]] # number of electrons (or allowed range)
    basis: List[BitDet] # list of bit determinants representing the basis states
     
    def __init__(self, norb: int, nelec: Optional[int | Sequence[int]]):
        """Initialize Fock space utilities with fixed number of orbitals and electrons.
        """
        if not isinstance(norb, int) or norb <= 0:
                raise ValueError("norb must be a positive integer")
                
        self.norb = norb
        self.nelec = nelec
        self._validate_nelec()
        self.basis = []
        self.generate_fock_basis()


    def _validate_nelec(self) -> None:
        """Validate that nelec is within valid range (0 <= nelec <= norb)."""
        if self.nelec is None:
            return
            
        if isinstance(self.nelec, int):
            if self.nelec < 0 or self.nelec > self.norb:
                raise ValueError(f"nelec must be between 0 and {self.norb}")
        else:
            for n in self.nelec:
                if n < 0 or n > self.norb:
                    raise ValueError(f"All nelec values must be between 0 and {self.norb}")

    @staticmethod
    def _popcount(x: int) -> int:
        """
        Counts the number of 1-bits in the binary representation of an integer.

        This function takes an integer as input and returns the count of
        bits that are set to 1 in its binary representation.

        :param x: An integer whose binary 1-bits are to be counted.
        :type x: int
        :return: The count of 1-bits in the binary representation of the input integer.
        :rtype: int
        """
        return x.bit_count()

    def _bits_before(self, state: int, orb: int) -> int:
        """
        Calculate the number of bits set to 1 in the binary representation of the input
        `state` up to (but not including) the given bit position `orb`.

        :param state: Integer representing the binary state from which the number of
            set bits will be counted.
        :type state: int
        :param orb: The bit position up to which the bits are considered (exclusive).
        :type orb: int
        :return: The count of bits set to 1 up to the specified position `orb`.
        :rtype: int
        """
        if orb < 0 or orb >= self.norb:
            raise ValueError(f"Orbital index {orb} out of range (0-{self.norb-1})")
        
        mask = (1 << orb) - 1
        return self._popcount(state & mask)
    
    def _validate_state(self, state: BitDet) -> None:
        """Check if state is valid for current norb."""
        if state < 0 or state >= (1 << self.norb):
            raise ValueError(f"State {state} invalid for {self.norb} orbitals")

    def apply_annihilation(self, state: BitDet, orb: int) -> Tuple[Optional[BitDet], int]:
        """
        Applies the annihilation operator to a given quantum state. The annihilation
        operator acts on a specific orbital in the state. If the orbital is not
        occupied, the function returns None and a phase of 0. Otherwise, it modifies
        the state by annihilating the particle in the specified orbital and computes
        the corresponding phase factor.

        :param state: The current quantum state represented as a bit determinant.
        :type state: BitDet

        :param orb: The orbital index on which the annihilation operator is applied.
        :type orb: int

        :return: A tuple where the first element is the modified quantum state
            represented as a bit determinant, or None if the orbital is not occupied,
            and the second element is the phase factor.
        :rtype: Tuple[Optional[BitDet], int]
        """
        self._validate_state(state)
        if orb < 0 or orb >= self.norb:
            raise ValueError(f"Orbital index {orb} out of range")
            
        if not (state >> orb & 1):
            return None, 0
        phase = (-1) ** self._bits_before(state, orb)
        return state & ~(1 << orb), phase

    def apply_creation(self, state: BitDet, orb: int) -> Tuple[Optional[BitDet], int]:
        """
        Apply a creation operator on the given quantum bit state.

        This function takes a quantum bit state (state) and applies a creation
        operator to the specified orbital (orb). If the orbital is already
        occupied (has a bit value of 1), the function will return None and a
        phase of 0. If the orbital is unoccupied (bit value of 0), the function
        calculates the resulting state after applying the creation operator and
        the associated phase factor.

        :param state: A quantum bit state represented as a `BitDet` type. Each bit
            represents the occupation state of an orbital.
        :param orb: The orbital index where the creation operator is applied.
            It must be a non-negative integer corresponding to the position of the
            bit in the quantum bit state.
        :return: A tuple containing the resulting quantum bit state after applying
            the creation operator and the associated phase factor. If the operation is
            invalid due to the orbital already being occupied, the function returns
            (None, 0).
        """
        self._validate_state(state)
        if orb < 0 or orb >= self.norb:
            raise ValueError(f"Orbital index {orb} out of range")
            
        if state >> orb & 1:
            return None, 0
        phase = (-1) ** self._bits_before(state, orb)
        return state | (1 << orb), phase

    def generate_fock_basis(self) -> None:
        """
        Generates the Fock basis based on the number of orbitals and a specified number
        of electrons. The function computes all possible bit determinants that meet
        the criteria for the given input parameters.

        :param norb: Number of orbitals. Must be a non-negative integer.
        :type norb: int
        :param n_elec: Number of electrons or a sequence of allowed numbers of
            electrons. If None, all possible configurations are considered.
            Can be an integer, a sequence of integers, or None.
        :type n_elec: Optional[int | Sequence[int]]
        :return: A list of bit determinants representing the Fock basis. Each bit
            determinant is an integer whose binary representation encodes the
            occupation of orbitals.
        """
        allowed = (
            set(range(self.norb + 1))
            if self.nelec is None
            else {self.nelec} if isinstance(self.nelec, int) else set(self.nelec)
        )
        self.basis = [det for det in range(1 << self.norb) if self._popcount(det) in allowed]

    def get_occupied_orbitals(self, state: int) -> list[int]:
        """Get list of occupied orbitals with validation."""
        self._validate_state(state)
        return [i for i in range(self.norb) if (state >> i) & 1]

    def create_state_from_occupied_orbitals(self, occupied_orbitals: list[int]) -> int:
        """Create state from occupied orbitals list with validation."""
        state = 0
        for orbital in occupied_orbitals:
            if orbital < 0 or orbital >= self.norb:
                raise ValueError(f"Orbital index {orbital} out of range")
            state |= (1 << orbital)
        return state

    def create_determinant_from_labels(self, all_labels: Dict[int, str], labels: List[str]) -> int:
        """Create determinant from orbital labels with full validation.
        
        :param all_labels: Dictionary mapping orbital indices (from 0 to norb-1) to their labels.
        :type all_labels: Dict[int, str]

        :param labels: List of orbital labels to be included in the determinant.
        :type labels: List[str]
        """
        if len(all_labels) != self.norb:
            raise ValueError(f"all_labels must contain exactly {self.norb} entries")
            
        state = 0
        particles_found = 0
        
        for label in labels:
            found = False
            for i, orb_label in all_labels.items():
                if label == orb_label:
                    if i < 0 or i >= self.norb:
                        raise ValueError(f"Label '{label}' maps to invalid orbital index {i}")
                    state |= (1 << i)
                    particles_found += 1
                    found = True
                    break
            if not found:
                raise ValueError(f"Label '{label}' not found in orbital labels")
        
        if self.nelec is not None:
            if isinstance(self.nelec, int):
                if particles_found != self.nelec:
                    raise ValueError(f"Expected {self.nelec} particles but found {particles_found}")
            elif particles_found not in self.nelec:
                raise ValueError(f"Particle count {particles_found} not in allowed set {self.nelec}")
        
        return state