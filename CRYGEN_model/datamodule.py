import random
from typing import Optional, Sequence, Union, Literal, Dict, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from .utils.data_utils import get_scaler_from_data_list, get_typed_scaler_from_data_list, preprocess, add_scaled_lattice_prop
#from dataset import CrystDataset
import pandas as pd
import os
from torch_geometric.data import Data

class CrystDataset(Dataset):
    def __init__(self, name: str, path: str,
                 prop_types: Dict[str, str], niggli: str, primitive: str,
                 graph_method: str, preprocess_workers: int,
                 lattice_scale_method: str, save_path: str, tolerance: float, use_space_group: bool, use_pos_index: bool,
                 prop_special_values: Dict[str, List[float]]={},
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop_types = prop_types
        self.prop_special_values = prop_special_values
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path, preprocess_workers)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_types=self.prop_types,
                prop_special_values=self.prop_special_values,
                use_space_group=self.use_space_group,
                tol=self.tolerance
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        # prop = self.scaler.transform(data_dict[self.prop])

        # assume ALL SCALAR!!!
        prop_dict = {}
        for prop, proptype in self.prop_types.items():
            if proptype == "binary":
                prop_dict[f"y_{prop}"] = torch.Tensor([data_dict[prop]]).long()
            elif proptype == "continuous":
                prop_dict[f"y_{prop}"] = torch.Tensor([data_dict[prop]]).view(1, -1).float()
            else:
                raise ValueError(f"Invalid property type: {proptype}")
        # print(prop_dict)
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            **prop_dict
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule:
    def __init__(
        self,
        dataset_configs: Dict[Literal['train', 'val', 'test'], Union[Dict, List[Dict]]],
        num_workers: Dict[Literal['train', 'val', 'test'], int],
        batch_size: Dict[Literal['train', 'val', 'test'], int],
        scaler_path=None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None
        self.dataset_configs = dataset_configs
        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:
            train_dataset = CrystDataset(**self.dataset_configs["train"])
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice')
            self.scaler = get_typed_scaler_from_data_list(
                train_dataset.cached_data,
                train_dataset.prop_types)
        else:
            try:
                self.lattice_scaler = torch.load(
                    Path(scaler_path) / 'lattice_scaler.pt')
                self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
            except:
                train_dataset = CrystDataset(**self.dataset_configs["train"])
                self.lattice_scaler = get_scaler_from_data_list(
                    train_dataset.cached_data,
                    key='scaled_lattice')
                self.scaler = get_typed_scaler_from_data_list(
                    train_dataset.cached_data,
                    train_dataset.prop_types)

    def setup(self, stage: Optional[Literal["fit", "test"]] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            self.train_dataset = CrystDataset(**self.dataset_configs["train"])
            if isinstance(self.dataset_configs["val"], dict):
                self.val_datasets = [
                    CrystDataset(**self.dataset_configs["val"])
                ]
            else:
                self.val_datasets = [
                    CrystDataset(**dataset_cfg)
                    for dataset_cfg in self.dataset_configs["val"]
                ]

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                CrystDataset(**dataset_cfg)
                for dataset_cfg in self.dataset_configs["test"]
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self, shuffle = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size["train"],
            num_workers=self.num_workers["train"],
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size["val"],
                num_workers=self.num_workers["val"],
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size["test"],
                num_workers=self.num_workers["test"],
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]
