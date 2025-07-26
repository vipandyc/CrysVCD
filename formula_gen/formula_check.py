import re
from tqdm import tqdm
import pandas as pd
from typing import Set
from pymatgen.core.composition import Composition
from collections import defaultdict
from . import PARENT_PATH
from .electronic_config import PURE_ALLOY_SPECIES, PURE_IONIC_SPECIES


ALLOY_ELEMENTS = set([e[:-1] for e in PURE_ALLOY_SPECIES])
element_symbol, valence_state = [], []
for species in PURE_IONIC_SPECIES:
    element_symbol.append(species[:-2])
    if species[-1] == "+":
        valence_state.append(int(species[-2]))
    elif species[-1] == "-":
        valence_state.append(-int(species[-2]))
VALENCES = pd.DataFrame({"element_symbol": element_symbol, "valence_state": valence_state})
EXCLUDED_ELEMENTS = set({'Tc': 43,
 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'He': 2,
 'Ne': 10, 'Ar': 18, 'Kr': 36, 'Xe': 54, 'Rn': 86, 'Po': 84, 'At': 85, 'Fr': 87,
 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103
}.keys())


def get_valence_symbol(element, valence_state):
    if valence_state > 0:
        return element + str(valence_state) + "+"
    elif valence_state == 0:
        return element + "0"
    else:
        return element + str(abs(valence_state)) + "-"


def check_valence_in_composition(composition, element, new_valence_state, sorted_valence_states):
    if len(sorted_valence_states) == 1:
        return True
    new_valence_state_index = sorted_valence_states.index(new_valence_state)
    if len(sorted_valence_states) == 2:
        another_valence_state = sorted_valence_states[1 - new_valence_state_index]
        if get_valence_symbol(element, another_valence_state) in composition:
            if another_valence_state * new_valence_state < 0:
                return False
        return True
    else:
        for i in range(len(sorted_valence_states)):
            if get_valence_symbol(element, sorted_valence_states[i]) in composition:
                if sorted_valence_states[i] * new_valence_state < 0:
                    return False
                if ((i < new_valence_state_index - 1) or (i > new_valence_state_index + 1)) and get_valence_symbol(element, sorted_valence_states[i]) in composition:
                    return False
        return True
    

def check_formula(element_dict):
    # alloy or single element
    if element_dict.keys() <= ALLOY_ELEMENTS:
        return {k + "0": v for k, v in element_dict.items()}, "A"
    
    # non-alloy
    valence_state_elements = {element: sorted(VALENCES[VALENCES["element_symbol"] == element]["valence_state"].tolist()) for element in element_dict.keys()}

    # naive formula
    dfs_leaves = [(defaultdict(int), 0)]
    new_dfs_leaves = []
    for i, (element, count) in enumerate(element_dict.items()):
        for composition, charge in dfs_leaves:
            valence_states = valence_state_elements[element]
            for valence_state in valence_states:
                if i == len(element_dict) - 1 and valence_state * charge > 0:
                    continue
            
                new_composition = composition.copy()
                new_composition[get_valence_symbol(element, valence_state)] += count
                new_dfs_leaves.append((new_composition, charge + valence_state * count))
        dfs_leaves = new_dfs_leaves
        new_dfs_leaves = []

    for composition, charge in dfs_leaves:
        if charge == 0:
            return composition, "I"

    # max negative charge
    max_negative_charge = 0
    for element, count in element_dict.items():
        valence_states = valence_state_elements[element]
        if valence_states:
            if valence_states[0] < 0:
                max_negative_charge += valence_states[0] * count

    # complex formula
    dfs_leaves = [(defaultdict(int), 0)]
    new_dfs_leaves = []
    for i, (element, count) in enumerate(element_dict.items()):
        for j in range(count):
            for composition, charge in dfs_leaves:
                if charge > -max_negative_charge:
                    continue
                valence_states = valence_state_elements[element]
                for valence_state in valence_states:
                    if i == len(element_dict) - 1 and valence_state * charge > 0:
                        continue
                    if check_valence_in_composition(composition, element, valence_state, valence_states):
                        new_composition = composition.copy()
                        new_composition[get_valence_symbol(element, valence_state)] += 1
                        dup_key = False
                        for leaf_composition, leaf_charge in new_dfs_leaves:
                            if new_composition.keys() == leaf_composition.keys():
                                dup_key = True
                                for k in new_composition.keys():
                                    if new_composition[k] != leaf_composition[k]:
                                        dup_key = False
                                        break
                        if not dup_key:
                            new_dfs_leaves.append((new_composition, charge + valence_state))
            dfs_leaves = new_dfs_leaves
            new_dfs_leaves = []

    for composition, charge in dfs_leaves:
        if charge == 0:
            return composition, "I"


def get_composition_with_valences(raw_formulas):
    composition_with_valences = []
    formula_types = []
    for composition in tqdm(raw_formulas):
        result = check_formula(composition)
        if result is not None:
            composition_with_valence, formula_type = result
            composition_with_valences.append(composition_with_valence)
            formula_types.append(formula_type)
        else:
            composition_with_valences.append(None)
            formula_types.append(None)
    return composition_with_valences, formula_types


def build_dataset():
    raw_formulas = pd.read_pickle(PARENT_PATH / "data" / "Chemical_formulas_raw.pkl")
    composition_with_valences, formula_types = get_composition_with_valences(raw_formulas["composition"])
    raw_formulas["composition_with_valence"] = composition_with_valences
    raw_formulas["formula_type"] = formula_types
    raw_formulas.to_pickle("Chemical_formulas_with_valence.pkl")
    raw_formulas[raw_formulas["formula_type"] == "I"].to_pickle(
        PARENT_PATH / "data" / "Chemical_formulas_with_valence_ionic.pkl"
    )
    raw_formulas[raw_formulas["formula_type"] == "A"].to_pickle(
        PARENT_PATH / "data" / "Chemical_formulas_with_valence_alloy.pkl"
    )


def extract_composition(cif_string):
    lines = cif_string.splitlines()
    for line in lines:
        if line.startswith("_chemical_formula_sum"):
            formula_line = line
            break
    if not formula_line:
        raise ValueError("No formula line found in the CIF string")

    # Extract the formula part after splitting on spaces
    chemical_formula = formula_line.split(maxsplit=1)[1].strip("'")

    # Parse the formula into a dictionary using regex
    parsed_formula = dict(re.findall(r"([A-Z][a-z]*)(\d*)", chemical_formula))
    parsed_formula = {element: int(count) if count else 1 for element, count in parsed_formula.items()}

    return parsed_formula    


def build_dataset_from_csv(csv_file, prop_types, prop_special_values, formula_type):
    df = pd.read_csv(csv_file)
    df["__checked__"] = True
    df["composition"] = [extract_composition(cif) for cif in df["cif"]]

    for i in range(len(df)):
        parsed_formula = df.loc[i, "composition"]
        if parsed_formula.keys() & EXCLUDED_ELEMENTS:
            df.loc[i, "__checked__"] = False
        for prop, prop_type in prop_types.items():
            if df.loc[i, prop] is pd.NA:
                df.loc[i, "__checked__"] = False
            elif prop_type == "continuous" and prop in prop_special_values:
                if type(prop_special_values[prop]) != list:
                    special_values = [prop_special_values[prop]]
                else:
                    special_values = prop_special_values[prop]
                for special_value in special_values:
                    if abs(df.loc[i, prop] - special_value) < 1e-6:
                        df.loc[i, "__checked__"] = False    
                        break

    cleaned_df = df[df["__checked__"]].copy()

    composition_with_valences, formula_types = get_composition_with_valences(cleaned_df["composition"])
    cleaned_df["composition_with_valence"] = composition_with_valences
    cleaned_df["formula_type"] = formula_types
    if formula_type == "alloy":
        final_df = cleaned_df[cleaned_df["formula_type"] == "A"]
    elif formula_type == "ionic":
        final_df = cleaned_df[cleaned_df["formula_type"] == "I"]
    else:
        raise ValueError(f"Invalid formula type: {formula_type}")
    print("length of final_df: ", len(final_df))
    return final_df


def read_formula_list(formula_list_file):
    with open(formula_list_file, "r") as f:
        formula_list = set(line.strip() for line in f.readlines() if line.strip())
    return formula_list


def check_duplicate_formula(composition: Composition, formula_set: Set[str], reduced: bool):
    if reduced:
        return composition.reduced_formula in formula_set
    else:
        return composition.formula in formula_set
