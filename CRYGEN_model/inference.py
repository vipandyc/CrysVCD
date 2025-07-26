import torch
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Union, Literal
from torch.utils.data import Dataset
import warnings, os
import chemparse, argparse
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from p_tqdm import p_map
from . import get_save_path
from .utils.eval_utils import lattices_to_params_shape, get_crystals_list
from .diffusion import CSPDiffusion  
from .CRYGEN_configs import get_config
warnings.filterwarnings("ignore")


chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

class SampleDataset(Dataset):

    def __init__(self, 
            datapoints: List[Dict[Literal["formula", "properties"], Union[str, Dict[str, Union[int, float]]]]],
            num_evals: int,\
            prop_types: Dict[str, Literal['binary', 'continuous']] = {'meta_stable': 'binary'},
        ):
        super().__init__()
        self.datapoints = datapoints
        self.num_evals = num_evals
        self.prop_types = prop_types
        self.preprocess()

    def preprocess(self):
        for dp in self.datapoints:
            formula = dp['formula']
            composition = chemparse.parse_formula(formula)
            chem_list = []
            for elem in composition:
                num_int = int(composition[elem])
                chem_list.extend([chemical_symbols.index(elem)] * num_int)
            dp['chem_list'] = chem_list

    def get_index(self, index):
        formula_index = index // self.num_evals
        replicate_index = index % self.num_evals
        return formula_index, replicate_index
        
    def __len__(self) -> int:
        return self.num_evals * len(self.datapoints)

    def __getitem__(self, index):
        formula_index, _ = self.get_index(index)
        prop_dict = dict()
        dp = self.datapoints[formula_index]
        for p, val in dp['properties'].items():
            if self.prop_types[p] == 'binary':
                prop_dict[f"y_{p}"] = torch.Tensor([val]).long()
            else:
                prop_dict[f"y_{p}"] = torch.Tensor([val]).view(1, -1)
        return Data(
            atom_types=torch.LongTensor(dp['chem_list']),
            num_atoms=len(dp['chem_list']),
            num_nodes=len(dp['chem_list']),
            **prop_dict
        )

def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None

def denoise(loader, model, step_lr):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []

    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )


def inference(save_path, config_name, formulas, prop_dict, num_evals, batch_size=4, step_lr=1e-5):
    config = get_config(config_name)
    training_config = config["training_config"]
    datamodule_config = config["datamodule_config"]
    prop_types = datamodule_config["dataset_configs"]["train"].get("prop_types", {"meta_stable": "binary"})
    model = CSPDiffusion(
        device=training_config["device"],
        prop_types=prop_types
    )
    model_path = get_save_path(training_config["save_path"], "model_checkpoints")
    inference_using = training_config.get("inference_using", "best")
    print("Loading model from ", model_path)
    model.load_state_dict(torch.load(model_path + f"_{inference_using}.pt"))
    

    if "guidance_path" in training_config is not None:
        guidance_path = get_save_path(training_config["guidance_path"], "model_checkpoints")
        print("Loading guidance model from ", guidance_path)
        guidance_model = CSPDiffusion(
            device=training_config["device"],
            prop_types=prop_types
        )
        guidance_using = training_config.get("guidance_using", "best")
        guidance_model.load_state_dict(torch.load(guidance_path + f"_{guidance_using}.pt"))
        model.load_guide_decoder(guidance_model)

    if torch.cuda.is_available():
        model.to('cuda')

    full_prop_dict = {'meta_stable': 1}
    full_prop_dict.update(prop_dict)

    if isinstance(formulas, str):
        datapoints = [{'formula': formulas, 'properties': full_prop_dict}]
    else:
        datapoints = [{'formula': f, 'properties': full_prop_dict} for f in formulas]

    test_set = SampleDataset(datapoints, num_evals=num_evals, prop_types=prop_types)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = denoise(test_loader, model, step_lr)

    print('Done inference!')

    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    strcuture_list = p_map(get_pymatgen, crystal_list)

    for i, structure in enumerate(strcuture_list):
        formula_index, replicate_index = test_set.get_index(i)
        formula = test_set.datapoints[formula_index]['formula']
        tar_dir = os.path.join(save_path, formula)
        os.makedirs(tar_dir, exist_ok=True)
        tar_file = os.path.join(tar_dir, f"{formula}_{replicate_index+1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i+1} Error Structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--formula', required=True)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_evals', default=2, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)

    args = parser.parse_args()


    inference(args.save_path, args.formula, args.num_evals, args.batch_size, args.step_lr)