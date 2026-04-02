import torch
from pymatgen.core import Structure, Composition
from pymatgen.ext.matproj import MPRester
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.io.ase import AseAtomsAdaptor
from mattersim.forcefield import MatterSimCalculator
# from ase.constraints import ExpCellFilter
from ase.optimize import GPMin
from ase.io import read
import os
from glob import glob
from tqdm import tqdm
import numpy as np

# Load a CIF file into pymatgen
def load_structure(cif_file):
    return Structure.from_file(cif_file)

# Compute energy per atom using ASE
def compute_energy_per_atom(structure):
    atoms = AseAtomsAdaptor.get_atoms(structure)
    
    # Set up MatterSim calculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mattersimcalc = MatterSimCalculator(device=device)
    atoms.calc = mattersimcalc

    # Relaxation
    # ecf = ExpCellFilter(atoms)
    dyn = GPMin(atoms)
    converged = dyn.run(fmax=0.03, steps=100)
    if not converged:
        return 'GEEE!'

    # Compute total energy and energy per atom
    total_energy = atoms.get_potential_energy()
    energy_per_atom = total_energy / len(atoms)
    
    return energy_per_atom

# Get phase diagram stability (Ehull) from Materials Project
def get_ehull(cif_file, api_key):
    structure = load_structure(cif_file)

    # Extract chemical elements
    elements = list(structure.composition.elements)
    chemsys = sorted([el.symbol for el in elements])  # Sorted order does not matter

    # Ensure the system is valid (<=4 elements)
    if len(chemsys) > 4:
        return -1.1 #"Too many elements (>4). Combinations not valid."
    
    atomic_numbers = [el.Z for el in structure.composition.elements]
    if max(atomic_numbers) > 83 or 43 in atomic_numbers or 61 in atomic_numbers:
        return -1.1
    
    noble_gas_flag = any(atom_number in atomic_numbers for atom_number in [2, 10, 18, 36, 54])
    if noble_gas_flag:
        return -1.1
    
    energy_per_atom = compute_energy_per_atom(structure)
    if energy_per_atom == 'GEEE!':
        # structure optimization not converged
        return 9.9 #'Structure_optimization not converged.'

    with MPRester(api_key) as m:
        entries = m.get_entries_in_chemsys(chemsys, compatible_only=True)
        for entry in entries:
            entry.energy_adjustments = []

    if not entries:
        return -1.1 #f"No phase diagram data available for {'-'.join(chemsys)}."

    # Create an entry for the given CIF file
    computed_entry = ComputedEntry(composition=Composition(structure.formula),
                                   energy=energy_per_atom * len(structure))

    # Construct phase diagram
    pd = PhaseDiagram(entries)
    ehull = pd.get_decomp_and_e_above_hull(computed_entry, allow_negative=True)

    return ehull[-1]

Ehull_vals_crygen = []
Ehull_vals_diffcsp = []

if __name__ == "__main__":
    API_KEY = "Yd2jih656GIoA2Ksfu57XrmOGh6seR2P"  # Replace with your Materials Project API key

    '''CIF_DIR = "../cif_tmp/crygen_cifs/"

    cif_files = glob(os.path.join(CIF_DIR, "*.cif"))
    
    for cif_file in tqdm(cif_files):
        try:
            ehull_value = get_ehull(cif_file, API_KEY)
        except Exception as e:
            print(e)
            ehull_value = 9.9
        print(f"{os.path.basename(cif_file)}: Energy above hull = {ehull_value} eV")
        Ehull_vals_crygen.append(ehull_value)'''
    
    CIF_DIR = "../../cif_tmp/cif_tmp_ehull005/"

    cif_files = glob(os.path.join(CIF_DIR, "*/*.cif"))
    
    for cif_file in tqdm(cif_files):
        try:
            ehull_value = get_ehull(cif_file, API_KEY)
        except Exception as e:
            print(e)
            ehull_value = 9.9
        print(f"{os.path.basename(cif_file)}: Energy above hull = {ehull_value} eV")
        Ehull_vals_diffcsp.append(ehull_value)

#np.save('crygen_ehull.npy', np.array(Ehull_vals_crygen))
np.save('crygen_ehull_005.npy', np.array(Ehull_vals_diffcsp))
