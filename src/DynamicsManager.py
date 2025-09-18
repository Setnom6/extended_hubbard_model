
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from qutip import Qobj, mesolve, Result

OneBodyDict = Dict[Tuple[int, int], complex]
TwoBodyDict = Dict[Tuple[int, int, int, int], complex]
BasesDict = Dict[str, Dict[str, List[str] | List[np.ndarray]]]




class DynamicsManager:
    def __init__(self):
        pass

    def performTimeEvolution(self, initial_state: Qobj, times: np.ndarray, 
                             precomputed_hamiltonians: Optional[List[Qobj]] = None,
                             unique_hamiltonian: Optional[Qobj] = None,
                             collapse_operators: Optional[List[Qobj]] = None) -> Result:
        """
        Function to perform a time evolution which is agnostic on the particular parameters and states

        Args:
            initial_state: np.ndarray, initial state vector in the Fock basis.
            times: np.ndarray, array of time points for the evolution.
            precomputed_hamiltonians: Optional[List[Qobj]], list of Hamiltonian matrices for each time step.
            unique_hamiltonian: Optional[Qobj], single Hamiltonian matrix if the Hamiltonian is time-independent.
            collapse_operators: Optional[List[Qobj]], list of collapse operators for Lindblad dynamics.
        """
        if collapse_operators is None:
            collapse_operators = []
        
        if precomputed_hamiltonians is not None:
            
            def H_t(t, args):
                # Linear interpolation of Hamiltonians between time steps as mesolve will compute more time points than the final result will show
                if t <= times[0]:
                    return precomputed_hamiltonians[0]
                if t >= times[-1]:
                    return precomputed_hamiltonians[-1]

                idx = np.searchsorted(times, t) - 1
                t0, t1 = times[idx], times[idx+1]
                H0, H1 = precomputed_hamiltonians[idx], precomputed_hamiltonians[idx+1]

                alpha = (t - t0) / (t1 - t0)
                H_interp = (1 - alpha) * H0.full() + alpha * H1.full()
                return Qobj(H_interp)


            result = mesolve(H_t, initial_state, times, collapse_operators, [], 
                         options={"nsteps": 10000, "atol":1e-7, "rtol": 1e-5, "method": 'bdf'})
            return result

        elif unique_hamiltonian is not None:
            result = mesolve(unique_hamiltonian, initial_state, times, collapse_operators, [], 
                         options={"nsteps": 10000, "atol":1e-7, "rtol": 1e-5, "method": 'bdf'})
            return result
        else:
            raise ValueError("Either precomputed_hamiltonians or unique_hamiltonian must be provided.")
        




    @staticmethod
    def build_protocol_timeline(listSlopes, totalPoints):
        """
        Build a generic protocol from a list of segments,
        ignoring segments with duration < 1e-10, and distributing
        totalPoints proportionally among the remaining segments.

        Args:
            listSlopes: list of lists, each element is [parameterInitialValue, parameterFinalValue, duration]
            totalPoints: int, total number of points in the concatenated protocol

        Returns:
            tlist: np.array, concatenated time list
            parameterValues: np.array, concatenated values of the varying parameter
        """
        # Filter out segments with negligible duration
        filteredSlopes = [seg for seg in listSlopes if seg[2] >= 1e-10]
        
        if not filteredSlopes:
            return np.array([]), np.array([])  # nothing to build

        # Total duration
        totalTime = sum(seg[2] for seg in filteredSlopes)
        
        # Initial number of points per segment (proportional to duration)
        rawPoints = [seg[2] * totalPoints / totalTime for seg in filteredSlopes]
        nValues = [int(rp) for rp in rawPoints]
        
        # Adjust to ensure the sum of points is exactly totalPoints
        pointsDiff = totalPoints - sum(nValues)
        # Distribute remaining points to segments with largest fractional part
        fractionalParts = [rp - int(rp) for rp in rawPoints]
        for idx in sorted(range(len(filteredSlopes)), key=lambda i: fractionalParts[i], reverse=True)[:pointsDiff]:
            nValues[idx] += 1
        
        tlistNano = []
        parameterValues = []
        tAcc = 0.0  # accumulated time
        
        for seg, nPoints in zip(filteredSlopes, nValues):
            tStart = tAcc
            tEnd = tAcc + seg[2]
            
            # Time array for this segment
            if nPoints == 1:
                tSegment = np.array([tStart])
            else:
                tSegment = np.linspace(tStart, tEnd, nPoints, endpoint=False)
            
            # Detuning array for this segment
            detStart, detEnd = seg[0], seg[1]
            if detStart == detEnd:
                eSegment = np.full(nPoints, detStart)
            else:
                eSegment = np.linspace(detStart, detEnd, nPoints, endpoint=False)
            
            tlistNano.append(tSegment)
            parameterValues.append(eSegment)
            
            tAcc = tEnd  # update accumulated time for next segment
        
        # Concatenate all segments
        tlistNano = np.concatenate(tlistNano)
        parameterValues = np.concatenate(parameterValues)
        
        return tlistNano, parameterValues
    

    @staticmethod
    def gamma_from_time(t_ns: float, nsToMeV: float) -> float:
        if abs(t_ns) < 1e-12:
            return None
        return 1.0 / t_ns / nsToMeV

    @staticmethod
    def decoherence_time(t2star_ns: float, t1_ns: float) -> float:
        if abs(t1_ns) < 1e-12 or abs(t2star_ns) < 1e-12:
            return 0.0
        return 1.0 / (1.0 / t2star_ns + 1.0 / (2.0 * t1_ns))