"""Legacy architectures preserved for checkpoint compatibility; CPU inference."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ForwardMember(nn.Module):
    def __init__(self, in_dim=20, out_dim=6, hidden_dims=(256,256,128), dropout=0.05):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ForwardEnsemble(nn.Module):
    def __init__(self, n_comp=15, n_proc=5, n_prop=6, n_members=5,
                 hidden_dims=(256,256,128), dropout=0.05):
        super().__init__()
        in_dim = n_comp + n_proc
        self.members = nn.ModuleList([ForwardMember(in_dim, n_prop, hidden_dims, dropout)
                                      for _ in range(n_members)])
        self.register_buffer('input_mean',  torch.zeros(in_dim))
        self.register_buffer('input_std',   torch.ones(in_dim))
        self.register_buffer('output_mean', torch.zeros(n_prop))
        self.register_buffer('output_std',  torch.ones(n_prop))
        self.register_buffer('_normalized', torch.tensor(0))

    def forward(self, composition, process):
        x = torch.cat([composition, process], dim=-1)
        if self._normalized.item() > 0:
            x = (x - self.input_mean.to(x.device)) / self.input_std.to(x.device)
        preds = torch.stack([m(x) for m in self.members], dim=0)
        if self._normalized.item() > 0:
            preds = preds * self.output_std.to(preds.device) + self.output_mean.to(preds.device)
        return preds.mean(0), preds.var(0)

class ElementEmbedding(nn.Module):
    def __init__(self, n_elements=15, embed_dim=32):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(n_elements, embed_dim)*0.02)
    def forward(self, comp): return comp @ self.embeddings

class SharedEncoder(nn.Module):
    def __init__(self, n_elements=15, n_properties=6, latent_dim=64, embed_dim=32, dropout=0.1):
        super().__init__()
        self.elem_embed = ElementEmbedding(n_elements, embed_dim)
        self.composition_mlp = nn.Sequential(
            nn.Linear(embed_dim,128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128,latent_dim), nn.LayerNorm(latent_dim))
        self.property_mlp = nn.Sequential(
            nn.Linear(n_properties,64), nn.GELU(),
            nn.Linear(64,latent_dim), nn.LayerNorm(latent_dim))
        self.cross_attn = nn.MultiheadAttention(latent_dim, 4, batch_first=True, dropout=dropout)
        self.fuse_norm  = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)
    def forward(self, composition, target_properties):
        z_S = self.composition_mlp(self.elem_embed(composition)).unsqueeze(1)
        z_P = self.property_mlp(target_properties).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=z_P, key=z_S, value=z_S)
        z = self.fuse_norm(z_S.squeeze(1) + z_P.squeeze(1) + attn_out.squeeze(1))
        return self.output_proj(z)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)].unsqueeze(0))

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, max_seq_len=5):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu')
        self.decoder     = nn.TransformerDecoder(layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
    def forward(self, tgt, memory):
        tgt = self.pos_encoder(tgt)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        return self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)

class MDNHead(nn.Module):
    def __init__(self, d_model=64, n_components=5):
        super().__init__()
        self.mu_net        = nn.Linear(d_model, n_components)
        self.log_sigma_net = nn.Linear(d_model, n_components)
        self.weight_logit_net = nn.Linear(d_model, n_components)
    def forward(self, h):
        mu     = torch.tanh(self.mu_net(h))
        sigma  = torch.exp(self.log_sigma_net(h).clamp(-4.0, -0.5))
        weights = F.softmax(self.weight_logit_net(h), dim=-1)
        return mu, sigma, weights
    def sample(self, h, temperature=1.0):
        mu, sigma, weights = self.forward(h)
        sigma = sigma * temperature
        g     = -torch.log(-torch.log(torch.rand_like(weights)+1e-8)+1e-8)
        comp  = F.softmax((torch.log(weights+1e-8)+g)/0.5, dim=-1)
        eps   = torch.randn_like(mu)
        s     = (comp * (mu + sigma*eps)).sum(-1, keepdim=True).clamp(-0.99, 0.99)
        lp    = self.log_prob(s, h)
        return s, lp
    def log_prob(self, value, h):
        mu, sigma, weights = self.forward(h)
        v = value.unsqueeze(-1) if value.dim()==1 else value
        diff = (v - mu) / (sigma+1e-8)
        lp   = (-0.5*diff**2 - torch.log(sigma+1e-8) - 0.5*torch.log(torch.tensor(2*math.pi, device=h.device)))
        return torch.logsumexp(torch.log(weights+1e-8) + lp, dim=-1)

class GumbelSoftmaxHead(nn.Module):
    def __init__(self, d_model=64, n_categories=4, hard=True, tau=0.5):
        super().__init__()
        self.n_cat = n_categories; self.hard = hard; self.tau = tau
        self.logit_net = nn.Linear(d_model, n_categories)
    def forward(self, h): return self.logit_net(h)
    def sample(self, h, temperature=None):
        tau    = temperature or self.tau
        logits = self.forward(h)
        y      = F.gumbel_softmax(logits, tau=tau, hard=self.hard)
        lp     = (y * F.log_softmax(logits, -1)).sum(-1)
        return y, lp

class AutoregressivePolicy(nn.Module):
    def __init__(self, n_elements=15, n_properties=6, latent_dim=64, d_model=64, n_components=5, n_quench_categories=4, decoder_layers=2, decoder_heads=4, dropout=0.1):
        super().__init__()
        self.encoder     = SharedEncoder(n_elements, n_properties, latent_dim, 32, dropout)
        self.decoder     = TransformerDecoder(d_model, decoder_heads, decoder_layers, d_model*4, dropout, 5)
        self.step_embed  = nn.Linear(1, d_model)
        self.sos_token   = nn.Parameter(torch.zeros(1,1,d_model))
        self.memory_proj = nn.Linear(latent_dim, d_model)
        self.head_delta_aus  = MDNHead(d_model, n_components)
        self.head_t_aus_time  = MDNHead(d_model, n_components)
        self.head_t_temper = MDNHead(d_model, n_components)
        self.head_t_temper_time = MDNHead(d_model, n_components)
        self.head_quench = GumbelSoftmaxHead(d_model, n_quench_categories)

    def _scale(self, v, lo, hi): return lo + (v+1)*0.5*(hi-lo)

    def forward(self, composition, target_properties, ac3, temperature=1.0):
        B = composition.shape[0]; dev = composition.device
        z      = self.encoder(composition, target_properties)
        memory = self.memory_proj(z).unsqueeze(1)
        tgt    = self.sos_token.expand(B,-1,-1)
        lp     = torch.zeros(B, device=dev)
        out    = []
        heads  = [self.head_delta_aus, self.head_t_aus_time, self.head_t_temper, self.head_t_temper_time]
        T_aus  = None

        for step in range(5):
            h = self.decoder(tgt, memory)[:,-1,:]
            if step == 0:
                v, l = heads[0].sample(h, temperature); v = v.squeeze(-1)
                delta = self._scale(v, 30., 150.)
                T_aus = (ac3 + delta).clamp(750., 1100.)
                out.append(T_aus.unsqueeze(-1))
                lp += l + math.log(2./120.)
                nxt = self.step_embed(T_aus.unsqueeze(-1))
            elif step == 1:
                v, l = heads[1].sample(h, temperature); v = v.squeeze(-1)
                p = self._scale(v, 0.5, 4.0)
                out.append(p.unsqueeze(-1))
                lp += l + math.log(2./3.5)
                nxt = self.step_embed(p.unsqueeze(-1))
            elif step == 2:
                q, l = self.head_quench.sample(h, temperature)
                idx  = q.argmax(-1, keepdim=True).float()
                out.append(idx)
                lp += l
                nxt = self.step_embed(idx)
            elif step == 3:
                v, l = heads[2].sample(h, temperature); v = v.squeeze(-1)
                t_max = (T_aus.clamp(max=720.) - 20.).clamp(min=170.)
                p = self._scale(v, 150., t_max)
                out.append(p.unsqueeze(-1))
                rng = (t_max - 150.).mean()
                lp += l + math.log(2./max(rng.item(), 1.))
                nxt = self.step_embed(p.unsqueeze(-1))
            elif step == 4:
                v, l = heads[3].sample(h, temperature); v = v.squeeze(-1)
                p = self._scale(v, 0.5, 8.0)
                out.append(p.unsqueeze(-1))
                lp += l + math.log(2./7.5)
                nxt = self.step_embed(p.unsqueeze(-1))
            tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
        return torch.cat(out, dim=-1), lp


# ─────────────────────────────────────────────────────────────────────────────
#  Physics & Analytics Engine
# ─────────────────────────────────────────────────────────────────────────────
def compute_ac3_np(C, Mn, Si, Ni, Cr, Mo, V, Cu):
    return (910. - 203.*np.sqrt(max(C, 1e-6)) - 15.2*Ni + 44.7*Si
            + 104.*V + 31.5*Mo - 30.*Mn - 11.*Cr - 20.*Cu)

def compute_ac3_torch(comp):
    C=comp[:,0]; Si=comp[:,1]; Mn=comp[:,2]; Cr=comp[:,5]
    Ni=comp[:,6]; Mo=comp[:,7]; V=comp[:,8]; Cu=comp[:,9]
    return (910. - 203.*torch.sqrt(C.clamp(min=1e-6)) - 15.2*Ni + 44.7*Si
            + 104.*V + 31.5*Mo - 30.*Mn - 11.*Cr - 20.*Cu)
