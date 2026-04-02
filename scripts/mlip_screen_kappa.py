import torch
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
from ase.io import read, write
from ase.optimize import BFGS
from ase import Atoms
from mattersim.forcefield import MatterSimCalculator
from phonopy.interface.calculator import read_crystal_structure
from phono3py import Phono3py
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
def setup_calculator(atoms: Atoms, device="cuda") -> Atoms:
    """Sets up the MatterSim calculator for the given Atoms object."""
    calc = MatterSimCalculator(device=device)
    atoms.calc = calc
    return atoms

def relax_structure(atoms):
    """Performs structure relaxation using BFGS."""
    dyn = BFGS(atoms)
    converged = dyn.run(fmax=0.001, steps=100)
    return atoms, converged

def compute_kappa(poscar_file):
    """Computes thermal conductivity for a given POSCAR file."""
    atoms = read(poscar_file)
    atoms = setup_calculator(atoms)

    # Relaxation
    atoms, converged = relax_structure(atoms)
    if not converged:
        print(f"Relaxation failed for {poscar_file}")
        return -99999
    # Save relaxed structure
    relaxed_poscar = "POSCAR-unitcell-temp"
    write(relaxed_poscar, atoms, format="vasp")

    # Load structure for phonon calculations
    unitcell, _ = read_crystal_structure(relaxed_poscar, interface_mode="vasp")

    ph3 = Phono3py(unitcell=unitcell, supercell_matrix=[2, 2, 2])
    ph3.generate_displacements()

    if len(ph3.supercells_with_displacements) > 5000:
        print(f"Number of supercells with displacements is too large: {len(ph3.supercells_with_displacements)}, skip this structure.")
        return None
    forces = []
    calculator = MatterSimCalculator(device=device)
    for i, sc in tqdm(enumerate(ph3.supercells_with_displacements), \
                    total=len(ph3.supercells_with_displacements), desc="Calculating forces", \
                    miniters=100):
        atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
        atoms.calc = calculator
        forces.append(atoms.get_forces())

    ph3.forces = np.array(forces)
    ph3.produce_fc2()
    ph3.produce_fc3()
    write_fc2_to_hdf5(ph3.fc2, filename="fc2.hdf5")
    write_fc3_to_hdf5(ph3.fc3, filename="fc3.hdf5")

    print(f'Start calculating kappa for {poscar_file}.')

    # Calculate thermal conductivity
    mesh = [15, 15, 15]  # Adjust as needed
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction(symmetrize_fc3q=True)
    temperature_range = np.arange(300, 301, 1)
    ph3.run_thermal_conductivity(temperatures=temperature_range, log_level=1)  # Only at 300K

    kappa = ph3.thermal_conductivity.kappa[0][0]
    kappa_avg = (kappa[0] + kappa[1] + kappa[2]) / 3  # Average over three directions

    print(f'Thermal conductivity for {poscar_file}: {kappa_avg}')
    return kappa_avg


INPUT_DIR = "../../cif_tmp/structures_for_kappa"  # Change this to the folder containing POSCAR files
OUTPUT_CSV = "mattersim_thermal_conductivity.csv"

# Initialize results DataFrame
results_df = pd.DataFrame(columns=["mp_id", "kappa"])

# Process all POSCAR files in the directory
for file in tqdm(os.listdir(INPUT_DIR), total=len(os.listdir(INPUT_DIR))):
    if file.startswith("POSCAR"):
        poscar_path = os.path.join(INPUT_DIR, file)

        # Extract mp_id using regex
        match = re.search(r"POSCAR_(mp-\d+)", file)
        if match:
            mp_id = match.group(1)
            print(mp_id)
        else:
            print(f"Skipping {file}: mp_id not found")
            continue

        # Compute kappa
        kappa_value = compute_kappa(poscar_path)
        if kappa_value is None:
            continue

        # Add result to DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([{"mp_id": mp_id, "kappa": kappa_value}])], ignore_index=True)
        results_df.to_csv(OUTPUT_CSV, index=False)


# Save results to CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to {OUTPUT_CSV}")