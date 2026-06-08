# MAST Inverse Design — Steel Heat Treatment Recommender (PRO VERSION)
# Production-ready Streamlit application featuring Object-Oriented architecture,
# interactive Plotly visualizations, native DOM cards, and state management.

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math, io, os, textwrap
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
#  Application Configuration & Constants
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAST Inverse Design | PRO",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ELEMENT_NAMES  = ['C','Si','Mn','P','S','Cr','Ni','Mo','V','Cu','Al','Ti','Nb','B','Fe']
PROPERTY_NAMES = ['UTS','YS','elongation','reduction_of_area','HRC','impact_energy_J']
QUENCH_MAP     = {0: 'Water', 1: 'Oil', 2: 'Air', 3: 'Polymer'}
PROP_UNITS     = {'UTS': 'MPa', 'YS': 'MPa', 'elongation': '%',
                  'reduction_of_area': '%', 'HRC': '', 'impact_energy_J': 'J'}

COMP_RANGES = {
    'C':(0.05,0.80), 'Si':(0.05,0.50), 'Mn':(0.30,2.00),
    'P':(0.001,0.04), 'S':(0.001,0.04), 'Cr':(0.00,2.00),
    'Ni':(0.00,4.00), 'Mo':(0.00,0.60), 'V':(0.00,0.30),
    'Cu':(0.00,0.50), 'Al':(0.005,0.06), 'Ti':(0.00,0.10),
    'Nb':(0.00,0.10), 'B':(0.000,0.005),
}

PROP_RANGES = {
    'UTS':(400,2000,1300), 'YS':(300,1800,1100),
    'elongation':(1,35,10), 'reduction_of_area':(3,40,15),
    'HRC':(15,65,40), 'impact_energy_J':(5,140,80),
}

STEEL_PRESETS = {
    'AISI 4340': dict(C=0.40,Si=0.25,Mn=0.75,P=0.02,S=0.02,Cr=0.80,Ni=1.80,Mo=0.25,V=0.001,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'AISI 1045': dict(C=0.45,Si=0.25,Mn=0.75,P=0.02,S=0.02,Cr=0.05,Ni=0.05,Mo=0.01,V=0.001,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'AISI 4140': dict(C=0.40,Si=0.25,Mn=0.85,P=0.02,S=0.02,Cr=0.95,Ni=0.05,Mo=0.20,V=0.001,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'H13 Tool':  dict(C=0.38,Si=1.00,Mn=0.40,P=0.02,S=0.02,Cr=5.15,Ni=0.05,Mo=1.35,V=1.00,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'Custom':    None,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Core PyTorch Model Architecture (Untouched for compatibility)
# ─────────────────────────────────────────────────────────────────────────────
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

@st.cache_resource(show_spinner="Initializing Model Weights into Memory...")
def bootstrap_models(fwd_bytes: Optional[bytes] = None, pol_bytes: Optional[bytes] = None):
    """Loads weights dynamically from either passed bytes or local paths."""
    device = torch.device('cpu')
    
    def _instantiate(fwd_ckpt, pol_ckpt):
        fwd = ForwardEnsemble().to(device)
        fwd.load_state_dict(fwd_ckpt['model_state_dict'])
        fwd.eval()

        actor = AutoregressivePolicy().to(device)
        actor.load_state_dict(pol_ckpt['actor_state_dict'])
        actor.eval()
        
        return fwd, actor, pol_ckpt.get('prop_mean'), pol_ckpt.get('prop_std'), device

    if fwd_bytes and pol_bytes:
        with io.BytesIO(fwd_bytes) as f1, io.BytesIO(pol_bytes) as f2:
            return _instantiate(
                torch.load(f1, map_location=device, weights_only=False),
                torch.load(f2, map_location=device, weights_only=False)
            )

    # Auto-load fallback
    fwd_path = next((p for p in [Path("forward_model.pt"), Path("checkpoints/forward_model.pt")] if p.exists()), None)
    pol_path = next((p for p in [Path("policy.pt"), Path("checkpoints/policy.pt")] if p.exists()), None)
    
    if fwd_path and pol_path:
        return _instantiate(
            torch.load(fwd_path, map_location=device, weights_only=False),
            torch.load(pol_path, map_location=device, weights_only=False)
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Professional UI Engine (Object-Oriented Structure)
# ─────────────────────────────────────────────────────────────────────────────
class SteelRecommenderPro:
    def __init__(self):
        self.initialize_state()
        self.inject_css()

    def initialize_state(self):
        # Ensures session variables persist across Streamlit re-runs.
        if 'models' not in st.session_state:
            st.session_state.models = bootstrap_models()
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'theme' not in st.session_state:
            st.session_state.theme = 'Auto'  # Auto | Light | Dark

    def inject_css(self):
        # Minimalist B&W theme — Auto follows OS, Light/Dark force a fixed palette.
        # Indigo brand accent on neutral zinc foundation.
        LIGHT_VARS = """
    --bg:        #fafafa;
    --surface:   #ffffff;
    --surface-2: #f4f4f5;
    --surface-3: #e4e4e7;
    --border:    #e4e4e7;
    --border-strong: #d4d4d8;
    --text:      #18181b;
    --text-muted:#52525b;
    --text-soft: #71717a;
    --accent:    #4f46e5;
    --accent-hover:#4338ca;
    --accent-soft:#eef2ff;
    --accent-text:#4338ca;
    --accent-inv:#ffffff;
    --success:   #059669;
    --success-soft:#d1fae5;
    --warning:   #d97706;
    --warning-soft:#fef3c7;
    --danger:    #dc2626;
    --danger-soft:#fee2e2;
    --info:      #2563eb;
    --info-soft: #dbeafe;
    --shadow-sm: 0 1px 2px rgba(24,24,27,0.05);
    --shadow:    0 4px 14px rgba(24,24,27,0.08);
    --shadow-lg: 0 14px 36px rgba(24,24,27,0.12);
    --shadow-accent: 0 6px 20px rgba(79,70,229,0.18);
    color-scheme: light;
"""
        DARK_VARS = """
    --bg:        #09090b;
    --surface:   #18181b;
    --surface-2: #27272a;
    --surface-3: #3f3f46;
    --border:    #27272a;
    --border-strong: #3f3f46;
    --text:      #fafafa;
    --text-muted:#a1a1aa;
    --text-soft: #71717a;
    --accent:    #6366f1;
    --accent-hover:#818cf8;
    --accent-soft:rgba(99,102,241,0.16);
    --accent-text:#a5b4fc;
    --accent-inv:#ffffff;
    --success:   #34d399;
    --success-soft:rgba(52,211,153,0.15);
    --warning:   #fbbf24;
    --warning-soft:rgba(251,191,36,0.15);
    --danger:    #f87171;
    --danger-soft:rgba(248,113,113,0.15);
    --info:      #60a5fa;
    --info-soft: rgba(96,165,250,0.15);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.4);
    --shadow:    0 4px 14px rgba(0,0,0,0.55);
    --shadow-lg: 0 14px 36px rgba(0,0,0,0.7);
    --shadow-accent: 0 6px 20px rgba(99,102,241,0.35);
    color-scheme: dark;
"""
        # Sync Streamlit's own theme variables to match.
        LIGHT_ST = """
    --primary-color: #4f46e5;
    --background-color: #fafafa;
    --secondary-background-color: #f4f4f5;
    --text-color: #18181b;
    --default-text-color: #18181b;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
"""
        DARK_ST = """
    --primary-color: #6366f1;
    --background-color: #09090b;
    --secondary-background-color: #27272a;
    --text-color: #fafafa;
    --default-text-color: #fafafa;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
"""
        SHARED = "--radius: 10px; --radius-sm: 6px;"
        LIGHT_VARS = LIGHT_VARS + LIGHT_ST
        DARK_VARS = DARK_VARS + DARK_ST

        theme = st.session_state.get('theme', 'Auto')
        if theme == 'Light':
            theme_block = f":root {{ {LIGHT_VARS}{SHARED} }}"
        elif theme == 'Dark':
            theme_block = f":root {{ {DARK_VARS}{SHARED} }}"
        else:  # Auto — light defaults, dark via OS media query
            theme_block = (
                f":root {{ {LIGHT_VARS}{SHARED} }}\n"
                f"@media (prefers-color-scheme: dark) {{ :root {{ {DARK_VARS} }} }}"
            )

        html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ───── Theme tokens ───── */
""" + theme_block + """

/* ───── Global base ───── */
html, body, [class*="css"], .stApp, .main, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg) !important;
    transition: background-color 0.25s ease, color 0.25s ease;
}
.main .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1280px; }

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
    font-weight: 700;
}
p, span, label, div, li { color: var(--text); }
a { color: var(--info) !important; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: var(--surface-2) !important; color: var(--text) !important;
       padding: 0.1rem 0.35rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }

/* ───── Sidebar — every internal node ───── */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"] {
    background-color: var(--surface) !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.1rem 2rem 1.1rem; }
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.25rem 0 !important;
}
[data-testid="stSidebarNav"] { background-color: var(--surface) !important; }
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] {
    color: var(--text) !important;
    background: var(--surface) !important;
}

/* ── Sidebar typography — single source of truth ──
   Two fonts only: Inter for prose/labels, JetBrains Mono for numeric values. */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] [data-baseweb] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text);
}

/* Inputs use mono so digits align. */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="input"] input {
    font-family: 'JetBrains Mono', 'Menlo', monospace !important;
    font-size: 0.875rem !important;
    color: var(--text) !important;
    background-color: var(--bg) !important;
    border-color: var(--border) !important;
    caret-color: var(--accent) !important;
}
[data-testid="stSidebar"] .stNumberInput [data-baseweb="input"],
[data-testid="stSidebar"] .stNumberInput [data-baseweb="input"] > div {
    background-color: var(--bg) !important;
}

/* Widget labels (e.g. "C (wt%)") — small uppercase muted.
   Scoped to NOT inside a radiogroup so the theme-toggle labels stay sentence case. */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stNumberInput label p,
[data-testid="stSidebar"] .stTextInput label p,
[data-testid="stSidebar"] .stSelectbox label p,
[data-testid="stSidebar"] .stSlider label p {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    line-height: 1.4 !important;
    margin-bottom: 0.3rem !important;
}

/* Selectbox display & dropdown chevron */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: var(--bg) !important;
    border-color: var(--border) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSidebar"] [data-baseweb="select"] [data-id="select"] {
    color: var(--text) !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] .stNumberInput button {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stSidebar"] .stNumberInput button:hover {
    background-color: var(--surface-3) !important;
}
[data-testid="stSidebar"] .stNumberInput button svg,
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: var(--text-muted) !important;
    fill: var(--text-muted) !important;
}

/* Slider min/max ticks and thumb value */
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
    background: transparent !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: var(--text) !important;
    background: transparent !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}

/* ───── Inputs ───── */
.stNumberInput input,
.stTextInput input,
.stTextArea textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.875rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    caret-color: var(--accent) !important;
}
.stNumberInput input:focus,
.stTextInput input:focus,
.stTextArea textarea:focus,
[data-baseweb="input"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent) !important;
    outline: none !important;
}
.stNumberInput [data-baseweb="input"] { background-color: var(--bg) !important; }
.stNumberInput button {
    background-color: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    transition: background 0.15s ease;
}
.stNumberInput button:hover {
    background-color: var(--surface-3) !important;
    border-color: var(--border-strong) !important;
}

label, .stNumberInput label, .stSelectbox label, .stSlider label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    font-size: 0.74rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem !important;
}

/* ───── Selectbox (trigger + dropdown popover) ───── */
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    min-height: 40px;
}
.stSelectbox [data-baseweb="select"] > div:hover { border-color: var(--border-strong) !important; }
.stSelectbox [data-baseweb="select"] svg { color: var(--text-muted) !important; }
[data-baseweb="select"] [role="combobox"] { color: var(--text) !important; }

/* Popover/dropdown menu — rendered at body root, can't be scoped to sidebar */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-lg) !important;
    padding: 0.3rem !important;
}
[data-baseweb="menu"] li,
[role="listbox"] [role="option"] {
    background-color: transparent !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-size: 0.875rem !important;
    padding: 0.45rem 0.65rem !important;
    transition: background 0.12s ease;
}
[role="listbox"] [role="option"]:hover,
[data-baseweb="menu"] li:hover,
[role="option"][aria-selected="true"] {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
}

/* ───── Sliders ───── */
.stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] > div { background: var(--surface-3) !important; }
.stSlider [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: var(--shadow-sm) !important;
    height: 16px !important; width: 16px !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: var(--text-soft) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ───── Buttons — primary (filled, gradient indigo) ───── */
div.stButton > button,
div.stDownloadButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.005em !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.7rem 1.5rem !important;
    transition: transform 0.16s ease, box-shadow 0.18s ease, filter 0.16s ease !important;
    box-shadow: var(--shadow-accent) !important;
    width: 100%;
    cursor: pointer;
    position: relative;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover,
div.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-accent), var(--shadow) !important;
    filter: brightness(1.05);
}
div.stButton > button:active { transform: translateY(0); filter: brightness(0.96); }
div.stButton > button:focus, div.stButton > button:focus-visible {
    outline: none !important;
    box-shadow: var(--shadow-accent), 0 0 0 3px color-mix(in srgb, var(--accent) 30%, transparent) !important;
}

/* Secondary button variant (Streamlit type="secondary") */
button[kind="secondary"] {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
}
button[kind="secondary"]:hover {
    background-color: var(--surface-2) !important;
    border-color: var(--text-muted) !important;
}

/* File uploader — full styling */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--surface-3) !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}
/* Inner "Browse files" button */
[data-testid="stFileUploader"] button,
[data-testid="stBaseButton-secondary"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.85rem !important;
    width: auto !important;
    box-shadow: none !important;
    transition: all 0.15s ease;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    background-color: var(--surface-2) !important;
    border-color: var(--text-muted) !important;
    transform: none !important;
}

/* ───── Tabs ───── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--surface);
    padding: 0.35rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 1.1rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg) !important;
    color: var(--accent-text) !important;
    box-shadow: var(--shadow-sm), inset 0 -2px 0 var(--accent) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ───── Cards ───── */
.mast-header {
    display: flex; flex-direction: column; gap: 0.35rem;
    padding-bottom: 1.5rem; margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}
.mast-header .eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--accent-text); letter-spacing: 0.15em; text-transform: uppercase;
    font-weight: 600;
    display: inline-flex; align-items: center; gap: 0.5rem;
}
.mast-header .eyebrow::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
}
.mast-header h1 {
    font-size: 2.4rem; margin: 0; font-weight: 700; letter-spacing: -0.035em;
}
.mast-header .subtitle {
    font-size: 0.95rem; color: var(--text-muted); margin: 0;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--text-muted);
    margin: 2rem 0 1rem 0;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-title::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}
/* In the sidebar, match the rest of the sidebar typography. */
[data-testid="stSidebar"] .section-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.6rem 0;
    color: var(--text-muted);
}

.pro-card-container {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem; margin: 1rem 0 2rem 0;
}
.pro-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.35rem;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.pro-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-hover));
}
.pro-card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: var(--shadow), 0 0 0 1px var(--accent-soft);
}
.pro-card-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem; font-weight: 600;
    color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.pro-card-rank {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.75rem; font-weight: 700;
    color: var(--accent); line-height: 1; margin-bottom: 0.15rem;
    letter-spacing: -0.02em;
}
.pro-card-score {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: var(--text-soft); padding-bottom: 0.9rem; margin-bottom: 0.9rem;
    border-bottom: 1px solid var(--border);
}
.pro-card-row {
    display: flex; justify-content: space-between; align-items: baseline;
    margin: 0.45rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.83rem;
}
.pro-label { color: var(--text-muted); font-weight: 500; }
.pro-value { color: var(--text); font-weight: 600; }
.pro-value.accent { color: var(--accent-text); }

/* Metric card (forward predictor) */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    transition: all 0.2s ease;
    position: relative;
}
.metric-card::after {
    content: ''; position: absolute; top: 0; right: 0;
    width: 32px; height: 3px; border-radius: 0 var(--radius) 0 0;
    background: linear-gradient(90deg, transparent, var(--accent));
    opacity: 0.5; transition: opacity 0.2s ease;
}
.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent-soft);
}
.metric-card:hover::after { opacity: 1; }
.metric-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700;
    color: var(--text); line-height: 1.1;
}
.metric-value .unit { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; margin-left: 0.25rem; }
.metric-std {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--text-soft); margin-top: 0.4rem;
}

/* Status banners */
.status-banner {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.85rem 1rem; border-radius: var(--radius-sm);
    margin-bottom: 1.25rem; border: 1px solid var(--border);
    background: var(--surface);
}
.status-banner.online { border-left: 3px solid var(--success); }
.status-banner.offline { border-left: 3px solid var(--danger); }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.status-dot.online { background: var(--success); box-shadow: 0 0 0 4px color-mix(in srgb, var(--success) 18%, transparent); }
.status-dot.offline { background: var(--danger); box-shadow: 0 0 0 4px color-mix(in srgb, var(--danger) 18%, transparent); }
.status-text {
    display: flex; flex-direction: column;
}
.status-text .label {
    font-size: 0.68rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}
.status-text .value {
    font-size: 0.9rem; color: var(--text); font-weight: 600;
}

/* Fe balance pill */
.fe-pill {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.75rem 1rem; border-radius: var(--radius-sm);
    background: var(--surface-2); border: 1px solid var(--border);
    margin-top: 1rem;
}
.fe-pill .fe-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.fe-pill .fe-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
}

/* Ac3 info chip */
.ac3-chip {
    display: inline-flex; gap: 0.55rem; align-items: baseline;
    padding: 0.6rem 0.95rem; border-radius: 999px;
    background: var(--accent-soft);
    border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
    margin: 1rem 0 1.5rem 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    color: var(--accent-text);
}
.ac3-chip strong { color: var(--accent-text); font-weight: 700; }
.ac3-chip .divider { color: var(--accent-text); opacity: 0.45; }

/* ───── Dataframe ───── */
.stDataFrame, [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
    background: var(--surface) !important;
}
[data-testid="stDataFrame"] [data-testid="stTable"] { background: var(--surface) !important; }
/* glide-data-grid (Streamlit's internal table renderer) */
.dvn-scroller, .glide-cell, [class*="dvn-"] {
    background-color: var(--surface) !important;
    color: var(--text) !important;
}

/* ───── Alerts (info/warning/error/success) ───── */
.stAlert, [data-testid="stAlert"], [data-testid="stNotification"] {
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    padding: 0.85rem 1rem !important;
}
.stAlert [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] p { color: var(--text) !important; font-size: 0.88rem; }
/* Variant tints via attribute */
[data-baseweb="notification"][kind="info"], .stAlert.st-warning { border-left: 3px solid var(--info) !important; }
.stAlert:has([data-testid="stAlertContentWarning"]),
.element-container .stAlert[data-baseweb*="warning"] { border-left: 3px solid var(--warning) !important; }
.stAlert:has([data-testid="stAlertContentError"]) { border-left: 3px solid var(--danger) !important; }
.stAlert:has([data-testid="stAlertContentSuccess"]) { border-left: 3px solid var(--success) !important; }
.stAlert:has([data-testid="stAlertContentInfo"]) { border-left: 3px solid var(--info) !important; }
/* SVG icons inside alerts */
.stAlert svg { color: var(--text-muted) !important; }

/* ───── Expander ───── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    background: var(--surface) !important;
    color: var(--text) !important;
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { background: var(--surface-2) !important; }

/* ───── Radio (non-toggle) & Checkbox ───── */
.stRadio [role="radiogroup"] label,
.stCheckbox label {
    color: var(--text) !important;
}
.stRadio [role="radio"][aria-checked="true"],
.stCheckbox [data-baseweb="checkbox"] [data-checked="true"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}
.stCheckbox [data-baseweb="checkbox"] > div:first-child {
    border-color: var(--border-strong) !important;
}

/* ───── Spinner ───── */
.stSpinner > div { border-color: var(--accent) transparent transparent transparent !important; }
.stSpinner + div { color: var(--text-muted) !important; font-size: 0.85rem; }

/* ───── Tooltip ───── */
[data-baseweb="tooltip"] {
    background: var(--text) !important;
    color: var(--bg) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 0.65rem !important;
}

/* ───── Plotly figure container ───── */
.js-plotly-plot, .plot-container {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 0.5rem 0.25rem;
}

/* ───── Hide Streamlit chrome ───── */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0 !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* ───── Sidebar brand block ───── */
.sidebar-brand {
    display: flex; align-items: center; gap: 0.65rem;
    margin: 0 0 0.5rem 0;
}
.brand-mark {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
    color: #ffffff;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700; font-size: 1.05rem;
    flex-shrink: 0;
    box-shadow: var(--shadow-accent);
    letter-spacing: -0.02em;
}
.brand-stack { display: flex; flex-direction: column; line-height: 1.15; }
.brand-title {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700; font-size: 0.98rem;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
.brand-subtitle {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem; font-weight: 600;
    color: var(--accent-text) !important;
    margin-top: 0.15rem;
    display: inline-flex; align-items: center;
    padding: 0.12rem 0.5rem;
    background: var(--accent-soft);
    border-radius: 999px;
    letter-spacing: 0.02em;
    width: fit-content;
}

/* ───── Appearance label (above theme toggle) ───── */
.appearance-label {
    display: flex; align-items: center; gap: 0.45rem;
    margin: 0.85rem 0 0.45rem 0;
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.appearance-label svg { opacity: 0.85; }

/* ───── Theme toggle — segmented pill ───── */
/* Targets the horizontal radio in the sidebar (Streamlit sets aria-orientation). */
[data-testid="stSidebar"] [role="radiogroup"][aria-orientation="horizontal"],
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 0.2rem !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.25rem !important;
    width: 100% !important;
    box-sizing: border-box;
}

[data-testid="stSidebar"] [role="radiogroup"][aria-orientation="horizontal"] > label,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {
    flex: 1 1 0 !important;
    margin: 0 !important;
    padding: 0.45rem 0.5rem !important;
    border-radius: 4px !important;
    cursor: pointer;
    transition: background 0.16s ease, color 0.16s ease, box-shadow 0.16s ease;
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    box-shadow: none !important;
    gap: 0 !important;
    min-height: 30px;
}

/* Hide the radio dot itself; we only want the label text */
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] ~ div,
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
}

/* Make the text span fill the label */
[data-testid="stSidebar"] [role="radiogroup"] > label [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [role="radiogroup"] > label > div {
    margin: 0 !important;
    color: inherit !important;
    width: 100%;
    text-align: center;
}
[data-testid="stSidebar"] [role="radiogroup"] > label p {
    color: inherit !important;
    margin: 0 !important;
    font-size: 0.78rem !important;
    font-weight: inherit !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    text-align: center;
    width: 100%;
}

[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    color: var(--text) !important;
    background: color-mix(in srgb, var(--surface) 60%, transparent) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked),
[data-testid="stSidebar"] [role="radiogroup"] > label[data-checked="true"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    box-shadow: var(--shadow-sm) !important;
    font-weight: 600 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-soft); }

/* ───── Responsive · Tablet (≤ 992px) ───── */
@media (max-width: 992px) {
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }
    .mast-header h1 { font-size: 1.95rem; }
    .mast-header .subtitle { font-size: 0.88rem; }
    .pro-card-container { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.85rem; }
    .pro-card { padding: 1.1rem; }
}

/* ───── Responsive · Mobile (≤ 640px) ───── */
@media (max-width: 640px) {
    .main .block-container {
        padding-top: 1rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        padding-bottom: 2rem !important;
    }

    /* Header scales */
    .mast-header {
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
    }
    .mast-header .eyebrow { font-size: 0.65rem; letter-spacing: 0.12em; }
    .mast-header h1 { font-size: 1.55rem; letter-spacing: -0.025em; }
    .mast-header .subtitle { font-size: 0.82rem; }

    /* Section titles compact */
    .section-title {
        font-size: 0.7rem; letter-spacing: 0.1em;
        margin: 1.5rem 0 0.75rem 0;
        gap: 0.4rem;
    }

    /* Force single column for st.columns on mobile */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 0 !important;
    }

    /* Process cards stack */
    .pro-card-container {
        grid-template-columns: 1fr;
        gap: 0.75rem;
        margin: 0.75rem 0 1.5rem 0;
    }
    .pro-card { padding: 1rem 1.1rem; }
    .pro-card-rank { font-size: 1.45rem; }
    .pro-card-row { font-size: 0.8rem; margin: 0.4rem 0; }

    /* Metric cards */
    .metric-card { padding: 0.9rem 1rem; margin-bottom: 0.65rem; }
    .metric-value { font-size: 1.35rem; }
    .metric-value .unit { font-size: 0.78rem; }

    /* Buttons — full width, larger touch target */
    div.stButton > button, div.stDownloadButton > button {
        padding: 0.85rem 1.25rem !important;
        font-size: 0.9rem !important;
        min-height: 44px;
    }

    /* Inputs — bigger tap targets, prevent iOS zoom-on-focus */
    .stNumberInput input, .stTextInput input, .stSelectbox > div > div {
        font-size: 16px !important;
        min-height: 42px;
    }
    .stNumberInput button { min-width: 36px; min-height: 36px; }

    /* Tabs — scroll horizontally if needed, slightly tighter */
    .stTabs [data-baseweb="tab-list"] {
        padding: 0.3rem;
        gap: 0.2rem;
        overflow-x: auto;
        flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.85rem !important;
        font-size: 0.82rem !important;
        white-space: nowrap;
    }

    /* Ac3 chip wraps gracefully */
    .ac3-chip {
        display: flex; flex-wrap: wrap; gap: 0.35rem;
        font-size: 0.78rem; padding: 0.55rem 0.75rem;
        margin: 0.75rem 0 1.25rem 0;
    }

    /* Status banner */
    .status-banner { padding: 0.75rem 0.85rem; gap: 0.65rem; }
    .status-text .label { font-size: 0.62rem; }
    .status-text .value { font-size: 0.85rem; }

    /* Fe pill */
    .fe-pill { padding: 0.65rem 0.85rem; }
    .fe-pill .fe-value { font-size: 0.95rem; }

    /* Sidebar trim on mobile drawer */
    [data-testid="stSidebar"] > div:first-child { padding: 1rem 0.85rem; }

    /* Dataframe horizontal scroll */
    [data-testid="stDataFrame"] { font-size: 0.78rem; }

    /* Plotly */
    .js-plotly-plot, .plot-container { padding: 0.25rem; }
}

/* ───── Responsive · Small phones (≤ 380px) ───── */
@media (max-width: 380px) {
    .mast-header h1 { font-size: 1.35rem; }
    .pro-card-rank { font-size: 1.25rem; }
    .metric-value { font-size: 1.2rem; }
    div.stButton > button { font-size: 0.85rem !important; padding: 0.8rem 1rem !important; }
}
</style>
"""
        st.markdown(html, unsafe_allow_html=True)


    def create_plotly_comparison(self, targets: Dict[str, float], results: List[Dict]) -> go.Figure:
        # Theme-neutral palette using Plotly's `template`-friendly transparent backgrounds.
        # Streamlit auto-injects font color via theme; we keep accents minimal.
        fig = go.Figure()
        props = PROPERTY_NAMES

        # Theme-neutral chrome; works against both light & dark page backgrounds.
        grid_color = 'rgba(113,113,122,0.18)'
        axis_color = 'rgba(113,113,122,0.45)'
        text_color = 'rgba(113,113,122,0.95)'

        # Target — outlined bars in indigo (brand accent).
        t_vals = [targets[p] for p in props]
        fig.add_trace(go.Bar(
            x=[f"{p}<br><span style='font-size:10px;opacity:0.6'>{PROP_UNITS[p]}</span>" for p in props],
            y=t_vals, name='Target',
            marker=dict(color='rgba(79,70,229,0.08)',
                        line=dict(color='#4f46e5', width=2)),
            hovertemplate='<b>Target</b><br>%{x}: %{y}<extra></extra>'
        ))

        # Sequential ramp: indigo → violet → fuchsia → teal-mint for ranks 1..5.
        # Single-hue dominant (indigo) with cool variation, keeping rank order legible.
        ramp = ['#4f46e5', '#6366f1', '#8b5cf6', '#0ea5e9', '#14b8a6']
        for i, r in enumerate(results):
            pred = [r['pred'][p] for p in props]
            err  = [r['std'][p] for p in props]
            fig.add_trace(go.Bar(
                x=[f"{p}<br><span style='font-size:10px;opacity:0.6'>{PROP_UNITS[p]}</span>" for p in props],
                y=pred, name=f"Rank {r['rank']}  ·  {r['score']:.2f}",
                marker=dict(color=ramp[i % len(ramp)],
                            line=dict(color='rgba(128,128,128,0.3)', width=0.5)),
                error_y=dict(type='data', array=err, visible=True,
                             color=axis_color, thickness=1.2, width=4),
                hovertemplate=f"<b>Rank {r['rank']}</b><br>%{{x}}: %{{y:.1f}} ± %{{error_y.array:.1f}}<extra></extra>"
            ))

        fig.update_layout(
            barmode='group', bargap=0.25, bargroupgap=0.08,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_color, family='Inter, sans-serif', size=12),
            xaxis=dict(showgrid=False, linecolor=axis_color, tickfont=dict(size=11)),
            yaxis=dict(showgrid=True, gridcolor=grid_color, linecolor=axis_color,
                       title=dict(text='Value', font=dict(size=11)), zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=40, b=20),
            hoverlabel=dict(font_family='JetBrains Mono', font_size=12),
        )
        return fig


    def render_sidebar(self):
        with st.sidebar:
            st.markdown(
                """
<div class="sidebar-brand">
    <div class="brand-mark">M</div>
    <div class="brand-stack">
        <div class="brand-title">MAST</div>
        <div class="brand-subtitle">Control Panel</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            # Theme toggle — Auto follows OS, Light / Dark force a palette.
            st.markdown(
                """
<div class="appearance-label">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="4"/>
        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
    </svg>
    <span>Appearance</span>
</div>
""",
                unsafe_allow_html=True,
            )
            options = ['Auto', 'Light', 'Dark']
            choice = st.radio(
                "Appearance",
                options,
                index=options.index(st.session_state.theme),
                horizontal=True,
                label_visibility='collapsed',
                key='theme_radio',
            )
            if choice != st.session_state.theme:
                st.session_state.theme = choice
                st.rerun()

            st.markdown("---")

            if st.session_state.models is not None:
                st.markdown(
                    """
<div class="status-banner online">
    <div class="status-dot online"></div>
    <div class="status-text">
        <span class="label">Engine Status</span>
        <span class="value">Online · Ready</span>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
<div class="status-banner offline">
    <div class="status-dot offline"></div>
    <div class="status-text">
        <span class="label">Engine Status</span>
        <span class="value">Offline — Upload weights</span>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
                fwd_file = st.file_uploader("Forward Model (.pt)", type=['pt'])
                pol_file = st.file_uploader("Policy Model (.pt)", type=['pt'])
                if fwd_file and pol_file:
                    st.session_state.models = bootstrap_models(fwd_file.read(), pol_file.read())
                    st.rerun()

            st.markdown('<div class="section-title">Composition</div>', unsafe_allow_html=True)
            preset = st.selectbox("Alloy Preset", list(STEEL_PRESETS.keys()))
            preset_vals = STEEL_PRESETS[preset]

            self.comp = {}
            for elem in ELEMENT_NAMES[:-1]:
                lo, hi = COMP_RANGES[elem]
                default = float(preset_vals[elem]) if preset_vals else (lo + hi) / 2
                self.comp[elem] = st.number_input(
                    f"{elem}  (wt%)", min_value=float(lo), max_value=float(hi),
                    value=float(np.clip(default, lo, hi)),
                    step=0.0001 if elem == 'B' else 0.01,
                    format="%.4f" if elem == 'B' else "%.3f",
                )

            fe_calc = 100.0 - sum(self.comp.values())
            self.comp['Fe'] = fe_calc

            fe_ok = 60 <= fe_calc <= 99.9
            fe_color = "var(--success)" if fe_ok else "var(--danger)"
            st.markdown(
                f"""
<div class="fe-pill">
    <span class="fe-label">Fe Balance</span>
    <span class="fe-value" style="color:{fe_color};">{fe_calc:.3f}%</span>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">Sampling</div>', unsafe_allow_html=True)
            self.n_cand = st.slider("Monte Carlo Candidates", 100, 1000, 300, 50)
            self.top_k  = st.slider("Top-K Display", 1, 10, 4)


    def execute_inference(self, targets: Dict[str, float]):
        # Runs the RL Policy and evaluates with the Forward Ensemble.
        fwd, actor, prop_mean, prop_std, device = st.session_state.models
        
        comp_arr = np.array([self.comp[e] for e in ELEMENT_NAMES], dtype=np.float32)
        tgt_arr = np.array([targets[p] for p in PROPERTY_NAMES], dtype=np.float32)
        
        comp_t = torch.tensor(comp_arr, device=device).unsqueeze(0).expand(self.n_cand, -1)
        tgt_t  = torch.tensor(tgt_arr, device=device).unsqueeze(0).expand(self.n_cand, -1)
        ac3 = compute_ac3_torch(comp_t)

        with torch.no_grad():
            processes, _ = actor(comp_t, tgt_t, ac3, temperature=0.6)
            P_hat, var   = fwd(comp_t, processes)

            if prop_mean is not None and prop_std is not None:
                pm, ps = prop_mean.to(device), prop_std.to(device)
                P_n, T_n = (P_hat - pm) / ps, (tgt_t - pm) / ps
            else:
                P_n, T_n = P_hat, tgt_t
                
            weights = torch.tensor([1.5, 1.5, 1.0, 0.8, 1.2, 0.8], device=device)
            mse = ((P_n - T_n)**2 * weights).sum(-1) / weights.sum()
            unc = (var / (prop_std**2 + 1e-8) if prop_std is not None else var).sum(-1)
            scores = -mse - 0.002 * unc

            top_idx = scores.argsort(descending=True)[:self.top_k]
            
            results = []
            for i, idx in enumerate(top_idx):
                proc = processes[idx].cpu().numpy()
                results.append({
                    'rank': i + 1, 'score': scores[idx].item(),
                    'T_aus': round(float(proc[0]), 1), 't_aus': round(float(proc[1]), 2),
                    'quench': QUENCH_MAP[int(proc[2])],
                    'T_temper': round(float(proc[3]), 1), 't_temper': round(float(proc[4]), 2),
                    'pred': dict(zip(PROPERTY_NAMES, P_hat[idx].cpu().numpy().tolist())),
                    'std': dict(zip(PROPERTY_NAMES, var[idx].sqrt().cpu().numpy().tolist())),
                })
        
        st.session_state.results = results


    def render_native_process_cards(self):
        if not st.session_state.results:
            return

        html = '<div class="pro-card-container">'
        for r in st.session_state.results:
            html += f"""<div class="pro-card">
    <div class="pro-card-header">Candidate</div>
    <div class="pro-card-rank">#{r['rank']}</div>
    <div class="pro-card-score">score · {r['score']:.3f}</div>
    <div class="pro-card-row"><span class="pro-label">Austenitize</span><span class="pro-value">{r['T_aus']} °C</span></div>
    <div class="pro-card-row"><span class="pro-label">Soak</span><span class="pro-value">{r['t_aus']} h</span></div>
    <div class="pro-card-row"><span class="pro-label">Quench</span><span class="pro-value accent">{r['quench']}</span></div>
    <div class="pro-card-row"><span class="pro-label">Temper</span><span class="pro-value">{r['T_temper']} °C</span></div>
    <div class="pro-card-row"><span class="pro-label">Hold</span><span class="pro-value">{r['t_temper']} h</span></div>
</div>"""
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


    def render_main(self):
        st.markdown(
            """
<div class="mast-header">
    <span class="eyebrow">MAST · Inverse Design Suite</span>
    <h1>Steel Heat-Treatment Recommender</h1>
    <p class="subtitle">Reinforcement-learning policy generator with forward-ensemble evaluation.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["Inverse Design", "Forward Predictor"])

        with tab1:
            if st.session_state.models is None:
                st.warning("Upload or place model weights (`forward_model.pt` & `policy.pt`) to initialize.")
                return

            st.markdown('<div class="section-title">Target Mechanical Properties</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            targets = {}
            for i, prop in enumerate(PROPERTY_NAMES):
                lo, hi, default = PROP_RANGES[prop]
                with cols[i % 3]:
                    targets[prop] = st.number_input(
                        f"{prop}  ({PROP_UNITS[prop]})",
                        min_value=float(lo), max_value=float(hi), value=float(default),
                        step=1.0 if lo > 5 else 0.1,
                    )

            ac3_est = compute_ac3_np(self.comp['C'], self.comp['Mn'], self.comp['Si'], self.comp['Ni'],
                                     self.comp['Cr'], self.comp['Mo'], self.comp['V'], self.comp['Cu'])
            st.markdown(
                f"""
<div class="ac3-chip">
    Ac<sub>3</sub> estimate <strong>{ac3_est:.1f} °C</strong>
    <span class="divider">·</span>
    austenitize window <strong>{ac3_est+30:.0f} – {ac3_est+150:.0f} °C</strong>
</div>
""",
                unsafe_allow_html=True,
            )

            run_col, _ = st.columns([1, 2])
            with run_col:
                if st.button("Synthesize Optimal Process"):
                    with st.spinner("Running policy rollouts…"):
                        self.execute_inference(targets)

            if st.session_state.results:
                st.markdown('<div class="section-title">Process Prescriptions</div>', unsafe_allow_html=True)
                self.render_native_process_cards()

                st.markdown('<div class="section-title">Predicted vs Target</div>', unsafe_allow_html=True)
                fig = self.create_plotly_comparison(targets, st.session_state.results)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                st.markdown('<div class="section-title">Detail Breakdown</div>', unsafe_allow_html=True)
                rows = []
                for r in st.session_state.results:
                    row = {'Rank': r['rank'], 'Score': f"{r['score']:.2f}",
                           'Quench': r['quench'], 'T_aus': r['T_aus'], 'T_temper': r['T_temper']}
                    for p in PROPERTY_NAMES:
                        row[p] = f"{r['pred'][p]:.1f} ± {r['std'][p]:.1f}"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index('Rank'), use_container_width=True)

        with tab2:
            if st.session_state.models is None:
                st.warning("Models offline. Cannot perform simulation.")
                return

            st.markdown('<div class="section-title">Simulation Parameters</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                t_aus_f = st.number_input("Austenitize Temp (°C)", 750., 1100., 870.)
                t_aus_t = st.number_input("Soak Time (h)", 0.5, 4.0, 1.0)
            with c2:
                q_f = st.selectbox("Quench Medium", ['Water', 'Oil', 'Air', 'Polymer'])
                q_i = {'Water': 0, 'Oil': 1, 'Air': 2, 'Polymer': 3}[q_f]
            with c3:
                t_tmp_f = st.number_input("Temper Temp (°C)", 150., 700., 450.)
                t_tmp_t = st.number_input("Temper Time (h)", 0.5, 8.0, 2.0)

            run_col, _ = st.columns([1, 2])
            with run_col:
                run_sim = st.button("Run Forward Simulation")

            if run_sim:
                fwd, _, _, _, device = st.session_state.models
                c_arr = np.array([self.comp[e] for e in ELEMENT_NAMES], dtype=np.float32)
                p_arr = np.array([t_aus_f, t_aus_t, q_i, t_tmp_f, t_tmp_t], dtype=np.float32)

                with torch.no_grad():
                    p_hat, var = fwd(torch.tensor(c_arr).unsqueeze(0), torch.tensor(p_arr).unsqueeze(0))

                pred, std = p_hat[0].numpy(), var[0].sqrt().numpy()
                st.markdown('<div class="section-title">Predicted Properties</div>', unsafe_allow_html=True)

                cols = st.columns(3)
                for i, prop in enumerate(PROPERTY_NAMES):
                    unit = PROP_UNITS[prop]
                    with cols[i % 3]:
                        st.markdown(
                            f"""
<div class="metric-card">
    <div class="metric-label">{prop}</div>
    <div class="metric-value">{pred[i]:.1f}<span class="unit">{unit}</span></div>
    <div class="metric-std">± {std[i]:.2f} uncertainty</div>
</div>
""",
                            unsafe_allow_html=True,
                        )


    def run(self):
        # Entry point mapping to execute the Application.
        self.render_sidebar()
        self.render_main()


if __name__ == "__main__":
    app = SteelRecommenderPro()
    app.run()