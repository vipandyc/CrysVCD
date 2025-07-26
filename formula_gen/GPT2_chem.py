import os
from itertools import permutations
from typing import Optional, Dict, Union, Literal, List, Tuple
import yaml
import numpy as np
import random
import torch
from tqdm import tqdm
from pymatgen.core.composition import Composition
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
from . import PARENT_PATH, IONIC_COMPOSITIONS_PATH, ALLOY_COMPOSITIONS_PATH
from .electronic_config import ALLOY_ELECTRONIC_CONFIG, IONIC_ELECTRONIC_CONFIG, ALLOY_SPECIES, IONIC_SPECIES, IONIC_VALENCES, get_element_valence
from .formula_check import read_formula_list, check_duplicate_formula

MAX_COUNT = 20  # max count of atoms of an element in a formula
MAX_LENGTH = 10  # max count of element types in a formula
TYPE_CONFIG = {
    "ionic": {
        "elements": IONIC_SPECIES,
        "electronic_config": IONIC_ELECTRONIC_CONFIG,
        "df_path": IONIC_COMPOSITIONS_PATH,
    },
    "alloy": {
        "elements": ALLOY_SPECIES,
        "electronic_config": ALLOY_ELECTRONIC_CONFIG,
        "df_path": ALLOY_COMPOSITIONS_PATH,
    },
}


def get_save_path(save_path, subfolder=""):
    # if save_path is not absolute, then it is relative to the parent path
    if not os.path.isabs(save_path):
        return str(PARENT_PATH / subfolder / save_path)
    else:
        return str(save_path)


class ElementList:
    def __init__(self, elements):
        self.elements = elements
        self.n_elements = len(elements)
        self.element_to_idx = {el: i for i, el in enumerate(elements)}
    
    def __getitem__(self, index):
        return self.elements[index]
    
    def __len__(self):
        return self.n_elements


class CompositionEmbedding(nn.Module):
    def __init__(self,
        dim_embedding: int,
        elements: ElementList, 
        use_electronic_config: bool,
        max_count: int,
        electronic_config: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.elements = elements
        self.max_count = max_count
        self.element_embedding = nn.Embedding(len(elements), dim_embedding)
        self.count_embedding = nn.Embedding(self.max_count + 1, dim_embedding)
        self.use_electronic_config = use_electronic_config
        if use_electronic_config:
            self.register_buffer("electronic_config", torch.tensor(electronic_config).float())
            self.electronic_layer = nn.Linear(self.electronic_config.shape[1], dim_embedding)
    
    def get_electronic_embedding(self, element_tokens: torch.LongTensor):
        return F.embedding(element_tokens, self.electronic_layer(self.electronic_config))

    def forward(self, element_tokens: torch.LongTensor, count_tokens: torch.LongTensor):
        element_embedding = self.element_embedding(element_tokens)
        count_embedding = self.count_embedding(count_tokens)
        composition_embedding = element_embedding + count_embedding
        if self.use_electronic_config:
            composition_embedding += self.get_electronic_embedding(element_tokens)
        return composition_embedding


class CompositionDataset(Dataset):
    def __init__(self, df, augmentation_fold: int, elements: ElementList, prop_types: dict):
        self.df = df
        self.data = []
        self.elements = elements
        self.augmentation_fold = augmentation_fold
        self.prop_types = prop_types
        self._process_data()

    def _process_data(self):
        print("Processing data " + ("and doing augmentation " if self.augmentation_fold > 1 else "") + "...")
        for _, row in tqdm(self.df.iterrows()):
            if row["composition_with_valence"] is None:
                continue
            comp_dict = row["composition_with_valence"]
            # Sort or reorder your dictionary keys if desired:
            elements = list(comp_dict.items())
            # e.g. [('Co',1), ('Mn',1), ('Na',3), ... ]
            perms = list(permutations(elements, len(elements)))
            elements_permutations = random.sample(perms, min(len(perms), self.augmentation_fold))

            for elements_perm in elements_permutations:
                # Build sequence of tokens, each is (element, count)
                tokens = [(self.elements.element_to_idx["START"], 0)] + \
                    [(self.elements.element_to_idx[el], ct) for el, ct in elements_perm] + \
                    [(self.elements.element_to_idx["END"], 0)]
                dp = {
                    "tokens": tokens,
                }
                for prop, prop_type in self.prop_types.items():
                    if prop_type == "continuous":
                        dp[prop] = torch.tensor([row[prop]]).float().reshape(-1, 1)
                    elif prop_type == "binary":
                        dp[prop] = torch.tensor([row[prop]]).long()
                    else:
                        raise ValueError(f"Invalid property type: {prop_type}")
                self.data.append(dp)
        if self.augmentation_fold > 1:
            print("Augmented data size: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def decode_logits(vector: torch.FloatTensor, elements: ElementList):
    num_elements = len(elements)
    el_part = vector[:num_elements]
    ct_part = vector[num_elements:]
    el_idx = el_part.argmax().item()
    element = elements[el_idx]
    count = ct_part.argmax().item()
    return element, count


def collate_fn(batch):
    prop_keys = batch[0].keys() - {"tokens"}
    all_input_elements = []
    all_input_counts = []
    all_target_elements = []
    all_target_counts = []
    all_masks = []
    all_props = {key: [] for key in prop_keys}
    
    for data_dict in batch:
        tokens = data_dict["tokens"]
        for key in prop_keys:
            all_props[key].append(data_dict[key])

        len_input = len(tokens) - 1
        elements, counts = zip(*tokens)
        input_elements = torch.tensor(elements[:-1])
        input_counts = torch.tensor(counts[:-1])
        target_elements = torch.tensor(elements[1:])
        target_counts = torch.tensor(counts[1:])
        
        # Pad to max_length if needed
        if len_input < MAX_LENGTH:
            pad_len = MAX_LENGTH - len_input
            input_mask = torch.cat([torch.ones(len_input), torch.zeros(pad_len)])
            input_elements = torch.cat([input_elements, torch.zeros(pad_len, dtype=torch.long)])
            input_counts = torch.cat([input_counts, torch.zeros(pad_len, dtype=torch.long)])
            target_elements = torch.cat([target_elements, torch.zeros(pad_len, dtype=torch.long)])
            target_counts = torch.cat([target_counts, torch.zeros(pad_len, dtype=torch.long)])
        else:
            input_mask = torch.ones(MAX_LENGTH)
        
        all_input_elements.append(input_elements)
        all_input_counts.append(input_counts)
        all_target_elements.append(target_elements)
        all_target_counts.append(target_counts)
        all_masks.append(input_mask)

    batch_input_elements = torch.stack(all_input_elements)  # (B, L)
    batch_input_counts = torch.stack(all_input_counts)
    batch_target_elements = torch.stack(all_target_elements)
    batch_target_counts = torch.stack(all_target_counts)
    batch_mask = torch.stack(all_masks)
    batch_props = {key: torch.concat(all_props[key], dim=0) for key in prop_keys} # (B,) for binary, (B, 1) for continuous
    return (
        batch_input_elements, batch_input_counts,
        batch_target_elements, batch_target_counts, 
        batch_mask, batch_props
    )


class GPT2ForChem(nn.Module):
    def __init__(self, 
        config: GPT2Config, 
        elements: ElementList, 
        electronic_config: np.ndarray, 
        max_count: int, 
        prop_types: Dict[str, Literal["continuous", "binary"]]={},
        verbose: bool=True
    ):
        super().__init__()
        self.model = GPT2Model(config)
        self.elements = elements
        self.embedding = CompositionEmbedding(
            dim_embedding=config.n_embd, 
            elements=elements, 
            use_electronic_config=True,
            max_count=max_count,
            electronic_config=electronic_config
        )
        self.prop_embedding = nn.ModuleDict()
        self.prop_types = prop_types
        for prop, prop_type in prop_types.items():
            if verbose:
                print(f"Building property embedding for {prop} with type {prop_type}")
            if prop_type == "continuous":
                self.prop_embedding[prop] = nn.Linear(1, config.n_embd)
            elif prop_type == "binary":
                self.prop_embedding[prop] = nn.Embedding(2, config.n_embd)
            else:
                raise ValueError(f"Invalid property type: {prop_type}")
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, 
        input_elements: torch.LongTensor, 
        input_counts: torch.LongTensor, 
        input_props: Dict[str, torch.Tensor], 
        attention_mask: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embedding = self.embedding(input_elements, input_counts) # (B, L, n_embd)
        for prop, prop_value in input_props.items():
            if prop in self.prop_embedding:
                input_embedding[:, 0, :] += self.prop_embedding[prop](prop_value)
        last_hidden_state = self.model(inputs_embeds=input_embedding, attention_mask=attention_mask).last_hidden_state
        outputs = self.lm_head(last_hidden_state)
        elem_logits = outputs[:, :, :len(self.elements)]
        count_logits = outputs[:, :, len(self.elements):]
        return elem_logits, count_logits
    

def predict_sequence(
    model: GPT2ForChem, 
    device: str, 
    elements: ElementList, 
    prop_dict: Dict[str, Union[float, int]]={}, 
    n_samples: int=10
) -> List[List[Tuple[str, int]]]:
    """
    Autoregressively predict the next tokens from 'start_tokens'
    until we reach 'END' or hit max_steps.
    """
    model.eval()
    collections = []
    with torch.no_grad():
        for _ in range(n_samples):
            predicted = [("START", 0)]
            input_props = {}
            for prop, prop_value in prop_dict.items():
                if prop not in model.prop_types:
                    continue
                if model.prop_types[prop] == "continuous":
                    input_props[prop] = torch.tensor(prop_value).float().reshape(-1, 1).to(device)
                elif model.prop_types[prop] == "binary":
                    input_props[prop] = torch.tensor(prop_value).long().to(device)
                else:
                    raise ValueError(f"Invalid property type: {model.prop_types[prop]}")
            for __ in range(MAX_LENGTH):
                # input shape (1, L_current)
                input_elements, input_counts = zip(*[(elements.element_to_idx[el], ct) for el, ct in predicted])
                input_elements = torch.tensor(input_elements).unsqueeze(0).to(device)
                input_counts = torch.tensor(input_counts).unsqueeze(0).to(device)
                
                elem_logits, count_logits = model(input_elements, input_counts, input_props)
                
                # Get the final step's prediction
                elem_logits = torch.softmax(elem_logits[0, -1, :], dim=-1)  # (num_elements,)
                count_logits = torch.softmax(count_logits[0, -1, :], dim=-1)  # (MAX_COUNT + 1,)
                
                elem_idx = torch.multinomial(elem_logits, 1).item()
                count_idx = torch.multinomial(count_logits, 1).item()
                elem = elements[elem_idx]
                count = count_idx

                if elem == "END":
                    break

                if count > 0 and elem != "PAD" and elem != "START":
                    predicted.append((elem, count))
            
            collections.append(predicted[1:])
    return collections


def get_config(formula_type, config_file, dataset_file=None):
    if not config_file:
        config_path = PARENT_PATH / "config" / f"GPT2_chem_{formula_type}.yaml"
    else:
        config_path = get_save_path(config_file, "config")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    elements = ElementList(TYPE_CONFIG[formula_type]["elements"])
    electronic_config = TYPE_CONFIG[formula_type]["electronic_config"]
    prop_types = config.get("prop_types", {})
    prop_special_values = config.get("prop_special_values", {})
    if dataset_file:
        df_path = get_save_path(dataset_file, "dataset")
        if df_path.endswith(".pkl"):
            df = pd.read_pickle(df_path)
        elif df_path.endswith(".csv"):
            from .formula_check import build_dataset_from_csv
            df = build_dataset_from_csv(df_path, prop_types, prop_special_values, formula_type)
        else:
            raise ValueError(f"Invalid dataset file: {dataset_file}")
    elif dataset_file is None:
        df = None
    else:
        df_path = TYPE_CONFIG[formula_type]["df_path"]
        if not os.path.exists(df_path):
            from .formula_check import build_dataset
            build_dataset()
        df = pd.read_pickle(df_path)
    return {
        "elements": elements,
        "electronic_config": electronic_config,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "gpt2_config": GPT2Config(
            vocab_size=len(elements) + MAX_COUNT + 1,
            n_positions=MAX_LENGTH,
            n_ctx=config["n_ctx"],
            n_embd=config["n_embd"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
        ),
        "augmentation_fold": config.get("augmentation_fold", 3),
        "df": df,
        "save_path": config["save_path"],
        "prop_types": prop_types,
        "pretrain_path": config.get("pretrain_path", None),
    }


class ValenceLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.valences = torch.tensor(IONIC_VALENCES, device=device).float() # (num_elements,)
        self.count = torch.tensor(range(MAX_COUNT + 1), device=device).float() # (MAX_COUNT + 1,)
        self.valence_criterion = nn.MSELoss()

    def forward(self, elem_logits, count_logits, mask):
        # elem_logits shape: (B, L, num_elements)
        # count_logits shape: (B, L, MAX_COUNT + 1)
        # elem_targets shape: (B, L)
        # count_targets shape: (B, L)
        # mask shape: (B, L)
        mean_elem_valence = torch.softmax(elem_logits, dim=-1) @ self.valences # (B, L)
        mean_count = torch.softmax(count_logits, dim=-1) @ self.count # (B, L)
        mean_tot_valence = torch.sum((mean_elem_valence * mean_count), dim=-1) / torch.sum(mean_count, dim=-1)
        return self.valence_criterion(mean_tot_valence, torch.zeros_like(mean_tot_valence))


def get_valence_error(sample):
    return sum([get_element_valence(el)[-1] * ct for el, ct in sample])


def train(formula_type: Literal["alloy", "ionic"], prop_dict: Dict[str, Union[float, int]]={}, config_file: str="", dataset_file: str=""):
    config = get_config(formula_type, config_file, dataset_file)
    device = config["device"]
    elements = config["elements"]
    electronic_config = config["electronic_config"]
    df = config["df"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = float(config["lr"])
    gpt2_config = config["gpt2_config"]
    augmentation_fold = config["augmentation_fold"]
    prop_types = config["prop_types"]
    if config["pretrain_path"] is None:
        pretrain_path = None
    else:
        pretrain_path = get_save_path(config["pretrain_path"], "checkpoint")

    dataset = CompositionDataset(df, 
        augmentation_fold=augmentation_fold, 
        elements=elements, 
        prop_types=prop_types
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = GPT2ForChem(gpt2_config, elements, electronic_config, MAX_COUNT, prop_types).to(device)
    if pretrain_path is not None:
        model.load_state_dict(torch.load(pretrain_path, weights_only=True), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elem_criterion = nn.CrossEntropyLoss(ignore_index=elements.element_to_idx["PAD"])
    count_criterion = nn.CrossEntropyLoss(ignore_index=0)
    valence_criterion = ValenceLoss(device)

    for epoch in range(epochs):
        model.train()
        total_loss, total_elem_loss, total_cnt_loss, total_valence_loss = 0.0, 0.0, 0.0, 0.0
        for (
            input_elements, input_counts, 
            target_elements, target_counts, 
            batch_mask, batch_props
        ) in tqdm(dataloader):
            # batch_inputs, batch_targets shape: (B, L)
            elem_logits, count_logits = model(
                input_elements.to(device), 
                input_counts.to(device), 
                {prop: prop_value.to(device) for prop, prop_value in batch_props.items()}, 
                batch_mask.to(device)
            )

            if formula_type == "ionic":
                loss_valence = valence_criterion(elem_logits, count_logits, batch_mask.to(device)) * 0
            else:
                loss_valence = torch.tensor(0.0, device=device)

            batch_mask = batch_mask.reshape(-1, 1)
            elem_logits = elem_logits.reshape(-1, elem_logits.size(-1))
            count_logits = count_logits.reshape(-1, count_logits.size(-1))
            elem_targets = target_elements.reshape(-1).to(device)
            count_targets = target_counts.reshape(-1).to(device)

            loss_elem  = elem_criterion(elem_logits, elem_targets)
            loss_count = count_criterion(count_logits, count_targets)
            
            loss = loss_elem + loss_count + loss_valence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cnt_loss += loss_count.item()
            total_elem_loss += loss_elem.item()
            total_valence_loss += loss_valence.item()
        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch [{epoch+1}/{epochs}] - ElemLoss: {total_elem_loss/len(dataloader):.4f}" + \
            f" - CntLoss: {total_cnt_loss/len(dataloader):.4f} " + (
                f"- ValenceLoss: {total_valence_loss/len(dataloader):.4f} " 
                if formula_type == "ionic" 
                else ""
            ) + f"- TotalLoss: {avg_loss:.4f}"
        )
        model.eval()
        with torch.no_grad():
            test_pred = predict_sequence(model, config["device"], config["elements"], prop_dict, n_samples=5)
            for sample in test_pred:
                print(sample, get_valence_error(sample))

    torch.save(model.state_dict(), get_save_path(config["save_path"], "checkpoint"))


def test(
    formula_type: Literal["alloy", "ionic"], 
    prop_dict: Dict[str, Union[float, int]], 
    n_samples: int = 10,
    config_file: str = "",
    dataset_file: str = "",
    verbose_flag: bool = True
) -> List[Dict[Literal["formula", "valence_error"], Union[str, float]]]:
    config = get_config(formula_type, config_file, dataset_file)
    model = GPT2ForChem(
        config["gpt2_config"], 
        config["elements"], 
        config["electronic_config"], 
        MAX_COUNT,
        prop_types=config["prop_types"],
        verbose=verbose_flag
    ).to(config["device"])
    model.load_state_dict(torch.load(get_save_path(config["save_path"], "checkpoint"), map_location=config["device"], weights_only=True))
    model.eval()
    test_pred = predict_sequence(model, config["device"], config["elements"], prop_dict, n_samples=n_samples)
    return [
        {
            "formula": "".join([get_element_valence(el)[0] + (
                str(ct) if ct > 1 else ""
            ) for el, ct in sample]),
            "valence_error": get_valence_error(sample)
        } for sample in test_pred
    ]


def generate_formula(
    formula_type: Literal["alloy", "ionic"], 
    n_formula: int, 
    prop_dict: Dict[str, Union[float, int]],
    config_file: str = "",
    dataset_file: str = "",
    exclude_formula: str = "none"
) -> List[str]:
    formulas = set()
    if exclude_formula == "formula":
        formula_list = read_formula_list("MP20_dataset/mp20_formula.txt")
        reduce_flag = False
    elif exclude_formula == "reduced_formula":
        formula_list = read_formula_list("MP20_dataset/mp20_reduced_formula.txt")
        reduce_flag = True
    verbose_flag = True
    tqdm_bar = tqdm(range(n_formula))
    while len(formulas) < n_formula:
        result = test(formula_type, prop_dict, 1, config_file, dataset_file, verbose_flag)[0]
        verbose_flag = False
        if not result["formula"]:
            continue
        if formula_type == "ionic" and result["valence_error"] != 0:
            continue
        result_formula = Composition(result["formula"])
        if exclude_formula != "none" and check_duplicate_formula(result_formula, formula_list, reduce_flag):
            continue
        formulas.add(result_formula.formula.replace(" ", ""))
        tqdm_bar.update(1)
    return list(formulas)
