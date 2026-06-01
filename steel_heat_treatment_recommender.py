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
import math, io, os
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

# High-Visibility Palette for the Plotly Graphs against the B&W Theme
RANK_COLORS = ['#00E5FF', '#D500F9', '#00E676', '#FFEA00', '#FF3D00']

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
    def __init__(self, n_comp=15, n_proc=5, n_prop=6, n_members=5, hidden_dims=(256,256,128), dropout=0.05):
        super().__init__()
        in_dim = n_comp + n_proc
        self.members = nn.ModuleList([ForwardMember(in_dim, n_prop, hidden_dims, dropout) for _ in range(n_members)])
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
    def forward(self, x): return self.dropout(x + self.pe[:x.size(1)].unsqueeze(0))

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

    fwd_path = next((p for p in [Path("forward_model.pt"), Path("checkpoints/forward_model.pt")] if p.exists()), None)
    pol_path = next((p for p in [Path("policy.pt"), Path("checkpoints/policy.pt")] if p.exists()), None)
    
    if fwd_path and pol_path:
        return _instantiate(
            torch.load(fwd_path, map_location=device, weights_only=False),
            torch.load(pol_path, map_location=device, weights_only=False)
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Professional UI Engine (Completely Overhauled Structural Layout)
# ─────────────────────────────────────────────────────────────────────────────
class SteelRecommenderPro:
    def __init__(self):
        self.inject_css()
        self.initialize_state()

    def initialize_state(self):
        if 'models' not in st.session_state:
            st.session_state.models = bootstrap_models()
        if 'results' not in st.session_state:
            st.session_state.results = None

    def inject_css(self):
        # Strict zero-indentation strings to absolutely prevent Streamlit Markdown <code> bugs
        css = ""
        css += "<style>\n"
        css += "@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');\n"
        css += "/* True B&W Foundation */\n"
        css += "html, body, [class*='css'], .stApp { font-family: 'Inter', sans-serif; background-color: #000000 !important; color: #FFFFFF !important; }\n"
        css += "h1, h2, h3, h4, h5, h6 { font-family: 'Space Mono', monospace !important; color: #FFFFFF !important; }\n"
        css += "/* Elegant Input Fields */\n"
        css += ".stNumberInput > div > div > input { background-color: #0D0D0D !important; color: #FFFFFF !important; border: 1px solid #333333 !important; font-family: 'Space Mono' !important; }\n"
        css += ".stSelectbox > div > div > div { background-color: #0D0D0D !important; color: #FFFFFF !important; border: 1px solid #333333 !important; }\n"
        css += "/* Tab Structure */\n"
        css += "[data-baseweb='tab-list'] { background-color: #0D0D0D; padding: 0.5rem; border-radius: 8px; border: 1px solid #222222; }\n"
        css += "[data-baseweb='tab'] { color: #888888; font-family: 'Space Mono', monospace; font-size: 0.9rem; }\n"
        css += "[aria-selected='true'] { background-color: #222222 !important; color: #FFFFFF !important; border-radius: 4px; }\n"
        css += "/* Custom Dashboard Containers */\n"
        css += ".dash-panel { background-color: #0A0A0A; border: 1px solid #333333; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }\n"
        css += ".dash-title { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #FFFFFF; border-bottom: 1px solid #333333; padding-bottom: 0.8rem; margin-bottom: 1.2rem; text-transform: uppercase; }\n"
        css += "/* Process Cards */\n"
        css += ".pro-card-wrapper { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }\n"
        css += ".pro-card { background: #0A0A0A; border: 1px solid #333333; border-radius: 6px; padding: 1.2rem; flex: 1; min-width: 220px; transition: transform 0.2s ease, border-color 0.2s ease; }\n"
        css += ".pro-card:hover { border-color: #FFFFFF; transform: translateY(-2px); }\n"
        css += ".pro-card-header { font-family: 'Space Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #FFFFFF; text-align: center; margin-bottom: 0.2rem; }\n"
        css += ".pro-card-score { text-align: center; color: #888888; font-size: 0.8rem; font-family: 'Space Mono', monospace; border-bottom: 1px dashed #333333; padding-bottom: 0.8rem; margin-bottom: 1rem; }\n"
        css += ".pro-card-row { margin: 0.6rem 0; font-family: 'Space Mono', monospace; font-size: 0.95rem; display: flex; justify-content: space-between; }\n"
        css += ".pro-label { color: #888888; font-weight: 700; }\n"
        css += ".pro-value { color: #FFFFFF; font-weight: 600; text-align: right; }\n"
        css += "/* Giant KPI Metrics for Forward Pass */\n"
        css += ".kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }\n"
        css += ".kpi-card { background: #0A0A0A; border: 1px solid #333333; padding: 1.5rem; border-radius: 6px; text-align: center; border-bottom: 3px solid #FFFFFF; }\n"
        css += ".kpi-title { font-family: 'Space Mono', monospace; color: #888888; font-size: 0.9rem; text-transform: uppercase; margin-bottom: 0.5rem; }\n"
        css += ".kpi-val { font-family: 'Space Mono', monospace; color: #FFFFFF; font-size: 2.2rem; font-weight: 700; }\n"
        css += ".kpi-unit { font-size: 1rem; color: #888888; }\n"
        css += ".kpi-err { font-family: 'Space Mono', monospace; color: #555555; font-size: 0.8rem; margin-top: 0.5rem; }\n"
        css += "/* Primary Button Overhaul */\n"
        css += "div.stButton > button { width: 100%; background-color: #FFFFFF !important; color: #000000 !important; border: none !important; font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; border-radius: 4px !important; padding: 0.8rem !important; transition: background-color 0.2s ease !important; }\n"
        css += "div.stButton > button:hover { background-color: #CCCCCC !important; }\n"
        css += "/* Subtle B&W Background Animation */\n"
        css += "#ambient-canvas { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; pointer-events: none; opacity: 0.08; }\n"
        css += "</style>\n"
        
        css += "<canvas id='ambient-canvas'></canvas>\n"
        css += "<script>\n"
        css += "const cvs = document.getElementById('ambient-canvas'); const ctx = cvs.getContext('2d');\n"
        css += "let w = cvs.width = window.innerWidth; let h = cvs.height = window.innerHeight;\n"
        css += "const pts = []; for (let i=0; i<60; i++) pts.push({x: Math.random()*w, y: Math.random()*h, vx: (Math.random()-0.5)*0.3, vy: (Math.random()-0.5)*0.3, r: Math.random()*1.2+0.5});\n"
        css += "window.addEventListener('resize', () => { w = cvs.width = window.innerWidth; h = cvs.height = window.innerHeight; });\n"
        css += "function draw() { ctx.clearRect(0,0,w,h); ctx.fillStyle='#FFFFFF'; ctx.lineWidth=0.5;\n"
        css += "for(let i=0; i<pts.length; i++) { let p = pts[i]; p.x += p.vx; p.y += p.vy; if(p.x<0||p.x>w) p.vx*=-1; if(p.y<0||p.y>h) p.vy*=-1;\n"
        css += "ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2); ctx.fill();\n"
        css += "for(let j=i+1; j<pts.length; j++) { let p2 = pts[j]; let d = Math.hypot(p.x-p2.x, p.y-p2.y); if(d<150) { ctx.strokeStyle=`rgba(255,255,255,${0.1*(1-d/150)})`; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p2.x, p2.y); ctx.stroke(); } } } requestAnimationFrame(draw); }\n"
        css += "draw();\n"
        css += "</script>\n"
        
        st.markdown(css, unsafe_allow_html=True)


    def create_plotly_comparison(self, targets: Dict[str, float], results: List[Dict]) -> go.Figure:
        fig = go.Figure()
        props = PROPERTY_NAMES
        
        # Highly visible Target Line
        t_vals = [targets[p] for p in props]
        fig.add_trace(go.Bar(
            x=[f"{p}<br>({PROP_UNITS[p]})" for p in props], y=t_vals, name='TARGET GOAL',
            marker=dict(color='rgba(0,0,0,0)', line=dict(color='#FFFFFF', width=2, dash='dot')), 
            hovertemplate='%{x}: %{y}<extra></extra>'
        ))

        # Vibrant colors against the dark layout to maximize readability
        for i, r in enumerate(results):
            pred = [r['pred'][p] for p in props]
            err = [r['std'][p] for p in props]
            color = RANK_COLORS[i % len(RANK_COLORS)]
            fig.add_trace(go.Bar(
                x=[f"{p}<br>({PROP_UNITS[p]})" for p in props], y=pred,
                name=f"RANK #{r['rank']} (Score: {r['score']:.1f})",
                marker_color=color,
                error_y=dict(type='data', array=err, visible=True, color='#FFFFFF', thickness=1.5),
                hovertemplate='%{x}: %{y:.1f} ± %{error_y.array:.1f}<extra></extra>'
            ))

        # Legend strictly forced to the bottom to prevent any overlap
        fig.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E6E6E6', family='Space Mono, monospace', size=12),
            xaxis=dict(showgrid=False, linecolor='#444444'),
            yaxis=dict(showgrid=True, gridcolor='#222222', linecolor='#444444', title='Metric Value'),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), # Crucial Overlap Fix
            margin=dict(l=40, r=40, t=20, b=80) 
        )
        return fig


    def render_sidebar(self):
        with st.sidebar:
            st.markdown("<h2 style='font-size: 1.4rem; letter-spacing: -1px; margin-bottom: 1.5rem;'>SYSTEM CONTROLS</h2>", unsafe_allow_html=True)
            
            if st.session_state.models is not None:
                st.markdown("<div style='border-left: 3px solid #00E676; padding: 0.5rem 1rem; background: #0A0A0A; border-radius: 4px; margin-bottom: 2rem;'><span style='color: #00E676; font-family: Space Mono; font-weight: bold; font-size: 0.9rem;'>● ENGINE ONLINE</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='border-left: 3px solid #FF3D00; padding: 0.5rem 1rem; background: #0A0A0A; border-radius: 4px; margin-bottom: 2rem;'><span style='color: #FF3D00; font-family: Space Mono; font-weight: bold; font-size: 0.9rem;'>○ ENGINE OFFLINE</span></div>", unsafe_allow_html=True)
                
                fwd_file = st.file_uploader("Inject Forward Model (.pt)", type=['pt'])
                pol_file = st.file_uploader("Inject Policy Model (.pt)", type=['pt'])
                if fwd_file and pol_file:
                    st.session_state.models = bootstrap_models(fwd_file.read(), pol_file.read())
                    st.rerun()

            st.markdown("<div style='font-family: Space Mono; font-size: 0.85rem; color: #888; text-transform: uppercase; margin-bottom: 0.5rem;'>Chemical Composition</div>", unsafe_allow_html=True)
            preset = st.selectbox("Load Standard Alloy", list(STEEL_PRESETS.keys()))
            preset_vals = STEEL_PRESETS[preset]

            self.comp = {}
            for elem in ELEMENT_NAMES[:-1]:
                lo, hi = COMP_RANGES[elem]
                default = float(preset_vals[elem]) if preset_vals else (lo+hi)/2
                self.comp[elem] = st.number_input(
                    f"{elem} (wt%)", min_value=float(lo), max_value=float(hi),
                    value=float(np.clip(default, lo, hi)),
                    step=0.0001 if elem == 'B' else 0.01, format="%.4f" if elem=='B' else "%.3f"
                )
                
            fe_calc = 100.0 - sum(self.comp.values())
            self.comp['Fe'] = fe_calc
            
            fe_color = '#00E676' if 60 <= fe_calc <= 99.9 else '#FF3D00'
            st.markdown(f"<div style='margin-top: 1rem; padding: 1rem; background: #0A0A0A; border: 1px solid #333;'><div style='color: #888; font-size: 0.75rem; font-family: Space Mono;'>IRON (Fe) BALANCE</div><div style='color: {fe_color}; font-size: 1.2rem; font-weight: bold; font-family: Space Mono;'>{fe_calc:.3f}%</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 2rem; font-family: Space Mono; font-size: 0.85rem; color: #888; text-transform: uppercase; margin-bottom: 0.5rem;'>Hyperparameters</div>", unsafe_allow_html=True)
            self.n_cand = st.slider("Monte Carlo Pool", 100, 1000, 300, 50)
            self.top_k  = st.slider("Output Ranks", 1, 10, 4)


    def execute_inference(self, targets: Dict[str, float]):
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
        # Strict zero-indentation inside the HTML string to prevent Markdown parsing bugs
        if not st.session_state.results: return
        
        html = ""
        html += "<div class='pro-card-wrapper'>\n"
        for i, r in enumerate(st.session_state.results):
            color = RANK_COLORS[i % len(RANK_COLORS)]
            html += f"<div class='pro-card' style='border-top: 3px solid {color};'>\n"
            html += f"<div class='pro-card-header'>RANK #{r['rank']}</div>\n"
            html += f"<div class='pro-card-score'>Score: {r['score']:.2f}</div>\n"
            html += f"<div class='pro-card-row'><span class='pro-label'>T_austenitize</span><span class='pro-value'>{r['T_aus']} °C</span></div>\n"
            html += f"<div class='pro-card-row'><span class='pro-label'>t_austenitize</span><span class='pro-value'>{r['t_aus']} hrs</span></div>\n"
            html += f"<div class='pro-card-row'><span class='pro-label'>Quenchant</span><span class='pro-value' style='color:{color};'>{r['quench']}</span></div>\n"
            html += f"<div class='pro-card-row'><span class='pro-label'>T_tempering</span><span class='pro-value'>{r['T_temper']} °C</span></div>\n"
            html += f"<div class='pro-card-row'><span class='pro-label'>t_tempering</span><span class='pro-value'>{r['t_temper']} hrs</span></div>\n"
            html += "</div>\n"
        html += "</div>\n"
        st.markdown(html, unsafe_allow_html=True)


    def render_main(self):
        st.markdown("<h1 style='font-size: 3rem; margin:0; letter-spacing: -2px; padding-top: 1rem;'>MAST<span style='color:#555;'>//</span>INVERSE</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: Space Mono; color: #888; font-size: 1rem; margin-bottom: 2rem;'>STEEL HEAT TREATMENT DESIGN WORKBENCH</p>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["[ 🎯 INVERSE DESIGN ]", "[ 🔬 FORWARD PREDICTOR ]"])

        with tab1:
            if st.session_state.models is None:
                st.info("System waiting for PyTorch weights. Please check the left panel.")
                return

            # Target Input Panel
            st.markdown("<div class='dash-panel'><div class='dash-title'>TARGET MECHANICAL PARAMETERS</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            targets = {}
            for i, prop in enumerate(PROPERTY_NAMES):
                lo, hi, default = PROP_RANGES[prop]
                with cols[i % 3]:
                    targets[prop] = st.number_input(
                        f"{prop} ({PROP_UNITS[prop]})",
                        min_value=float(lo), max_value=float(hi), value=float(default),
                        step=1.0 if lo > 5 else 0.1
                    )
            
            ac3_est = compute_ac3_np(self.comp['C'], self.comp['Mn'], self.comp['Si'], self.comp['Ni'],
                                     self.comp['Cr'], self.comp['Mo'], self.comp['V'], self.comp['Cu'])
            st.markdown(f"<div style='margin-top: 1.5rem; color: #666; font-family: Space Mono; font-size: 0.8rem;'>Calculated Ac3: <span style='color:#FFF;'>{ac3_est:.1f} °C</span> | Operating Bounds: {ac3_est+30:.0f}°C – {ac3_est+150:.0f}°C</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Central Action Button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("SYNTHESIZE OPTIMAL PROCESS (DEEP RL)"):
                with st.spinner("Executing Policy Gradient Search..."):
                    self.execute_inference(targets)

            if st.session_state.results:
                st.markdown("<br><hr style='border-color: #222;'><br>", unsafe_allow_html=True)
                
                # Visual Dashboard Area
                st.markdown("<div class='dash-title' style='border:none; text-align:center;'>PERFORMANCE VS TARGET METRICS</div>", unsafe_allow_html=True)
                fig = self.create_plotly_comparison(targets, st.session_state.results)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                st.markdown("<div class='dash-title' style='border:none; text-align:center; margin-top:2rem;'>THEORETICAL PROCESS PRESCRIPTIONS</div>", unsafe_allow_html=True)
                self.render_native_process_cards()

                with st.expander("VIEW RAW DATA MATRIX"):
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
                st.info("System waiting for PyTorch weights. Please check the left panel.")
                return
                
            st.markdown("<div class='dash-panel'><div class='dash-title'>SIMULATION PARAMETERS</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                t_aus_f = st.number_input("Austenitize Temp (°C)", 750., 1100., 870.)
                t_aus_t = st.number_input("Soak Time (hrs)", 0.5, 4.0, 1.0)
            with c2:
                q_f = st.selectbox("Quench Medium", ['Water','Oil','Air','Polymer'])
                q_i = {'Water':0,'Oil':1,'Air':2,'Polymer':3}[q_f]
            with c3:
                t_tmp_f = st.number_input("Temper Temp (°C)", 150., 700., 450.)
                t_tmp_t = st.number_input("Temper Time (hrs)", 0.5, 8.0, 2.0)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("EXECUTE FORWARD PASS"):
                fwd, _, _, _, device = st.session_state.models
                c_arr = np.array([self.comp[e] for e in ELEMENT_NAMES], dtype=np.float32)
                p_arr = np.array([t_aus_f, t_aus_t, q_i, t_tmp_f, t_tmp_t], dtype=np.float32)
                
                with torch.no_grad():
                    p_hat, var = fwd(torch.tensor(c_arr).unsqueeze(0), torch.tensor(p_arr).unsqueeze(0))
                
                pred, std = p_hat[0].numpy(), var[0].sqrt().numpy()
                
                st.markdown("<br><hr style='border-color: #222;'><br>", unsafe_allow_html=True)
                st.markdown("<div class='dash-title' style='border:none; text-align:center;'>PREDICTED MATERIAL PROPERTIES</div>", unsafe_allow_html=True)
                
                html_metrics = "<div class='kpi-grid'>\n"
                for i, prop in enumerate(PROPERTY_NAMES):
                    html_metrics += f"<div class='kpi-card'>\n"
                    html_metrics += f"<div class='kpi-title'>{prop}</div>\n"
                    html_metrics += f"<div class='kpi-val'>{pred[i]:.1f} <span class='kpi-unit'>{PROP_UNITS[prop]}</span></div>\n"
                    html_metrics += f"<div class='kpi-err'>± {std[i]:.1f} Uncertainty</div>\n"
                    html_metrics += "</div>\n"
                html_metrics += "</div>\n"
                
                st.markdown(html_metrics, unsafe_allow_html=True)


    def run(self):
        self.render_sidebar()
        self.render_main()


if __name__ == "__main__":
    app = SteelRecommenderPro()
    app.run()