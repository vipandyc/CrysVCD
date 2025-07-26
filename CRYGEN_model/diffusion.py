import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Any, Dict, Literal
from tqdm import tqdm
from .cspnet import CSPNet
from .utils.diff_utils import BetaScheduler, SigmaScheduler, d_log_p_wrapped_normal
from .utils.data_utils import lattice_params_to_matrix_torch


MAX_ATOMIC_NUM=100


### Model definition
class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(nn.Module):
    def __init__(self, 
        time_dim=256, 
        latent_dim=0, 
        cost_coord=1.0, 
        cost_lattice=1.0, 
        max_neighbors=12, 
        radius=5.0, 
        timesteps=1000,
        device='cuda',
        prop_dim=256,
        prop_types={'meta_stable':'binary'}
    ) -> None:
        super().__init__()

        self.decoder = CSPNet(
            hidden_dim = 512,
            latent_dim = time_dim + prop_dim,
            max_atoms = 100,
            num_layers = 6,
            act_fn = 'silu',
            dis_emb = 'sin',
            num_freqs = 128,
            edge_style = 'fc',
            max_neighbors = max_neighbors,
            cutoff = radius,
            ln = True,
            ip = True
        )
        self.guide_decoder = None
        self.prop_types = prop_types
        self.prop_adapter = nn.ModuleDict()
        # at most two properties.
        for prop, proptype in self.prop_types.items():
            print("Building property adapter for", prop, proptype)
            if proptype == 'binary':
                self.prop_adapter[prop] = nn.Embedding(2, prop_dim)
            elif proptype == 'continuous':
                self.prop_adapter[prop] = nn.Linear(1, prop_dim)
            else:
                raise KeyError(f"Invalid property type: {proptype}")

        self.beta_scheduler = BetaScheduler(
            scheduler_mode="cosine",
            timesteps = timesteps
        )
        self.sigma_scheduler = SigmaScheduler(
            timesteps = timesteps,
            sigma_begin = 0.005,
            sigma_end = 0.5
        )
        self.time_dim = time_dim
        self.prop_dim = prop_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.keep_lattice = cost_lattice < 1e-5
        self.keep_coords = cost_coord < 1e-5
        self.device = device
        self.drop_prob = 0.25

    def load_guide_decoder(self, guidance_model: "CSPDiffusion"):
        self.guide_decoder = guidance_model.decoder

    def forward(self, batch: Batch) -> Dict[Literal['loss', 'lattice_loss', 'coord_loss'], torch.Tensor]:
        batch_size = batch.num_graphs
        
        # create property embedding
        prop_emb = torch.zeros(batch.num_graphs, self.prop_dim, device=self.device)
        for prop, proptype in self.prop_types.items():
            prop_vals = batch[f"y_{prop}"]
            prop_module = self.prop_adapter[prop]  # Retrieve the corresponding module
            prop_embedding_tmp = prop_module(prop_vals)  # Pass the input through the module
            prop_emb = prop_emb + prop_embedding_tmp
        # prop_emb: (256, 256)

        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        # time_emb: (256, 256)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices
        # pass in graph score network, here classifier-free guidance use Bernoulli to dropout
        keep_mask = torch.bernoulli(torch.ones(batch_size) * self.drop_prob).bool().to(self.device)

        # Apply mask to remove selected rows
        prop_emb[keep_mask] = 0

        # KONGLONG MAMA TIBI WANGZI
        pred_l, pred_x = self.decoder(time_emb, prop_emb, batch.atom_types, \
                                      input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = (
            self.cost_lattice * loss_lattice +
            self.cost_coord * loss_coord
        )

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5, guide_w=2):

        batch_size = batch.num_graphs
        
        # create property embedding
        prop_emb = torch.zeros(batch.num_graphs, self.prop_dim, device=self.device)
        zero_emb = torch.zeros(batch.num_graphs, self.prop_dim, device=self.device)
        for prop, proptype in self.prop_types.items():
            prop_vals = batch[f"y_{prop}"]
            prop_module = self.prop_adapter[prop]  # Retrieve the corresponding module
            prop_embedding_tmp = prop_module(prop_vals)  # Pass the input through the module
            prop_emb = prop_emb + prop_embedding_tmp

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            # classifier-free guidance!
            pred_l_con, pred_x_con = self.decoder(time_emb, prop_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            if self.guide_decoder is not None:
                pred_l_uncon, pred_x_uncon = self.guide_decoder(time_emb, zero_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            else:
                pred_l_uncon, pred_x_uncon = self.decoder(time_emb, zero_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x, pred_l = (1 + guide_w) * pred_x_con - guide_w * pred_x_uncon, (1 + guide_w) * pred_l_con - guide_w * pred_l_uncon

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            # classifier-free guidance!
            pred_l_con, pred_x_con = self.decoder(time_emb, prop_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            if self.guide_decoder is not None:
                pred_l_uncon, pred_x_uncon = self.guide_decoder(time_emb, zero_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            else:
                pred_l_uncon, pred_x_uncon = self.decoder(time_emb, zero_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x, pred_l = (1 + guide_w) * pred_x_con - guide_w * pred_x_uncon, (1 + guide_w) * pred_l_con - guide_w * pred_l_uncon

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> Dict[Literal['loss', 'lattice_loss', 'coord_loss'], torch.Tensor]:
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        if loss.isnan():
            return None

        return {
            "loss" : loss,
            "lattice_loss" : loss_lattice,
            "coord_loss" : loss_coord
        }

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[Literal['val_loss', 'val_lattice_loss', 'val_coord_loss'], torch.Tensor]:

        output_dict = self(batch)

        log_dict = self.compute_stats(output_dict, prefix='val')

        return log_dict

    def test_step(self, batch: Any, batch_idx: int) -> Dict[Literal['test_loss', 'test_lattice_loss', 'test_coord_loss'], torch.Tensor]:

        output_dict = self(batch)

        log_dict = self.compute_stats(output_dict, prefix='test')

        return log_dict

    def compute_stats(self, output_dict: Dict[Literal['loss', 'lattice_loss', 'coord_loss'], torch.Tensor], prefix: Literal['val', 'test']):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict
    