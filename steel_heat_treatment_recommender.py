"""
MAST Inverse Design — Steel Heat Treatment Recommender (PRO VERSION)
Production-ready Streamlit application featuring Object-Oriented architecture,
interactive Plotly visualizations, native DOM cards, and state management.
"""

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
        self.inject_css()
        self.initialize_state()

    def initialize_state(self):
        """Ensures session variables persist across Streamlit re-runs."""
        if 'models' not in st.session_state:
            st.session_state.models = bootstrap_models()
        if 'results' not in st.session_state:
            st.session_state.results = None

    def inject_css(self):
        """Injects enterprise-grade, responsive CSS and the ambient Canvas animation."""
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

        /* Global Resets */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #030303 !important;
            color: #FFFFFF !important;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 { font-family: 'Space Mono', monospace !important; color: #FFFFFF !important; }

        /* Background Canvas */
        #ambient-canvas {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            z-index: -1; pointer-events: none; opacity: 0.12;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0a0a0a !important; border-right: 1px solid #222 !important;
        }

        /* Native HTML Process Cards */
        .pro-card-container {
            display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; margin-bottom: 2rem;
        }
        .pro-card {
            background: linear-gradient(145deg, #0d0d0d, #050505);
            border: 1px solid #333; border-radius: 6px; padding: 1.5rem;
            flex: 1; min-width: 220px; transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .pro-card:hover {
            border-color: #777; transform: translateY(-3px); box-shadow: 0 8px 25px rgba(255,255,255,0.05);
        }
        .pro-card-header {
            font-family: 'Space Mono', monospace; font-size: 1.2rem; font-weight: 700;
            color: #FFF; text-align: center; margin-bottom: 0.3rem;
        }
        .pro-card-score {
            text-align: center; color: #888; font-size: 0.8rem; font-family: 'Space Mono', monospace;
            border-bottom: 1px dashed #333; padding-bottom: 0.8rem; margin-bottom: 1rem;
        }
        .pro-card-row { margin: 0.5rem 0; font-family: 'Space Mono', monospace; font-size: 0.9rem; }
        .pro-label { color: #666; width: 85px; display: inline-block; font-weight: 700; }
        .pro-value { color: #E6E6E6; }

        /* Unified Buttons */
        div.stButton > button {
            background-color: #FFFFFF !important; color: #000000 !important;
            border: none !important; font-family: 'Space Mono', monospace;
            font-weight: 700; border-radius: 4px !important; padding: 0.6rem 2rem !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button:hover {
            background-color: #cccccc !important; box-shadow: 0 0 15px rgba(255,255,255,0.3);
        }
        </style>
        
        <!-- Ambient Neural Network Canvas -->
        <canvas id="ambient-canvas"></canvas>
        <script>
        const canvas = document.getElementById('ambient-canvas');
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;
        const numPoints = 75; const points = [];
        for (let i = 0; i < numPoints; i++) {
            points.push({ x: Math.random() * width, y: Math.random() * height,
                          vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4,
                          radius: Math.random() * 1.5 + 1 });
        }
        window.addEventListener('resize', () => { width = canvas.width = window.innerWidth; height = canvas.height = window.innerHeight; });
        function animate() {
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.lineWidth = 0.8;
            for (let i = 0; i < numPoints; i++) {
                const p = points[i]; p.x += p.vx; p.y += p.vy;
                if (p.x < 0 || p.x > width) p.vx *= -1; if (p.y < 0 || p.y > height) p.vy *= -1;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2); ctx.fill();
                for (let j = i + 1; j < numPoints; j++) {
                    const p2 = points[j]; const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (dist < 160) {
                        ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 * (1 - dist / 160)})`;
                        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(animate);
        }
        animate();
        </script>
        """, unsafe_allow_html=True)


    def create_plotly_comparison(self, targets: Dict[str, float], results: List[Dict]) -> go.Figure:
        """Generates a highly professional, interactive, monochrome Plotly graph."""
        fig = go.Figure()
        props = PROPERTY_NAMES
        
        # Target Line (Hollow bars)
        t_vals = [targets[p] for p in props]
        fig.add_trace(go.Bar(
            x=[f"{p}<br>({PROP_UNITS[p]})" for p in props], y=t_vals, name='Target Goal',
            marker=dict(color='rgba(0,0,0,0)', line=dict(color='#FFFFFF', width=2)),
            hovertemplate='%{x}: %{y}<extra></extra>'
        ))

        # Prediction Bars with Grayscale mapping
        shades = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333']
        for i, r in enumerate(results):
            pred = [r['pred'][p] for p in props]
            err = [r['std'][p] for p in props]
            fig.add_trace(go.Bar(
                x=[f"{p}<br>({PROP_UNITS[p]})" for p in props], y=pred,
                name=f"Rank #{r['rank']} (Score: {r['score']:.1f})",
                marker_color=shades[i % len(shades)],
                error_y=dict(type='data', array=err, visible=True, color='#FF4444' if i==0 else '#FFFFFF', thickness=1.5),
                hovertemplate='%{x}: %{y:.1f} ± %{error_y.array:.1f}<extra></extra>'
            ))

        fig.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E6E6E6', family='Space Mono, monospace', size=11),
            title=dict(text='TARGET vs. ENSEMBLE PREDICTIONS', font=dict(color='#FFFFFF', size=16)),
            xaxis=dict(showgrid=False, linecolor='#444'),
            yaxis=dict(showgrid=True, gridcolor='#222', linecolor='#444', title='Metric Value'),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=80, b=20)
        )
        return fig


    def render_sidebar(self):
        """Constructs the sidebar for model injection and chemical constraints."""
        with st.sidebar:
            st.markdown("### `SYSTEM COMPONENT` ⚙️")
            st.markdown("---")
            
            # Status Banner
            if st.session_state.models is not None:
                st.markdown("""
                <div style="border: 1px solid #2ECC71; padding: 0.8rem; background-color: #051005; border-radius: 4px; margin-bottom:1rem;">
                    <p style="margin:0; font-size:0.75rem; color:#888; font-family:'Space Mono';">ENGINE STATUS</p>
                    <p style="margin:0; font-size:0.95rem; font-weight:bold; color:#2ECC71; font-family:'Space Mono';">✓ ONLINE & READY</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="border: 1px dashed #E74C3C; padding: 0.8rem; background-color: #100505; border-radius: 4px; margin-bottom:1rem;">
                    <p style="margin:0; font-size:0.75rem; color:#888; font-family:'Space Mono';">ENGINE STATUS</p>
                    <p style="margin:0; font-size:0.95rem; font-weight:bold; color:#E74C3C; font-family:'Space Mono';">✗ OFFLINE</p>
                    <p style="margin:0.2rem 0 0 0; font-size:0.75rem; color:#aaa;">Upload model weights below.</p>
                </div>
                """, unsafe_allow_html=True)
                
                fwd_file = st.file_uploader("Forward Model (.pt)", type=['pt'])
                pol_file = st.file_uploader("Policy Model (.pt)", type=['pt'])
                if fwd_file and pol_file:
                    st.session_state.models = bootstrap_models(fwd_file.read(), pol_file.read())
                    st.rerun()

            st.markdown("**STEEL CHEMICAL COMPOSITION**")
            preset = st.selectbox("Alloy Preset", list(STEEL_PRESETS.keys()))
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
            
            # Iron safety indicator
            st.markdown(f"""
            <div style="padding: 0.5rem; background: #111; border: 1px solid #333; margin-top: 1rem;">
                <span style="font-family:'Space Mono'; font-size:0.8rem; color:#aaa;">Fe Balance:</span><br>
                <span style="font-family:'Space Mono'; font-size:1.1rem; font-weight:bold; color:{'#2ECC71' if 60<=fe_calc<=99.9 else '#E74C3C'}">{fe_calc:.3f}%</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            self.n_cand = st.slider("Monte Carlo Candidates", 100, 1000, 300, 50)
            self.top_k  = st.slider("Display Candidates Count", 1, 10, 4)


    def execute_inference(self, targets: Dict[str, float]):
        """Runs the RL Policy and evaluates with the Forward Ensemble."""
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
        """Builds beautiful HTML cards for recommendations avoiding static Matplotlib images."""
        if not st.session_state.results: return
        
        html_blocks = ['<div class="pro-card-container">']
        for r in st.session_state.results:
            html_blocks.append(f"""
            <div class="pro-card">
                <div class="pro-card-header">RANK #{r['rank']}</div>
                <div class="pro-card-score">Score: {r['score']:.2f}</div>
                <div class="pro-card-row"><span class="pro-label">T_aus:</span> <span class="pro-value">{r['T_aus']} °C</span></div>
                <div class="pro-card-row"><span class="pro-label">t_aus:</span> <span class="pro-value">{r['t_aus']} hrs</span></div>
                <div class="pro-card-row"><span class="pro-label">Quench:</span> <span class="pro-value" style="color: #FFF;">{r['quench']}</span></div>
                <div class="pro-card-row"><span class="pro-label">T_temp:</span> <span class="pro-value">{r['T_temper']} °C</span></div>
                <div class="pro-card-row"><span class="pro-label">t_temp:</span> <span class="pro-value">{r['t_temper']} hrs</span></div>
            </div>
            """)
        html_blocks.append('</div>')
        st.markdown("\n".join(html_blocks), unsafe_allow_html=True)


    def render_main(self):
        """Constructs the primary user interface tabs and triggers."""
        st.markdown("""
        <div style="border-bottom: 1px solid #333; padding-bottom: 1rem; margin-bottom: 2rem; margin-top: 1rem;">
            <h1 style="font-size: 2.5rem; margin:0; letter-spacing:-1px;">MAST · Inverse Design</h1>
            <h3 style="font-size: 1rem; color: #888; margin:0; font-weight: normal;">Production Grade Policy Generator & Forward Evaluator</h3>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🎯 INVERSE DESIGN WORKSPACE", "🔬 FORWARD PREDICTOR"])

        with tab1:
            if st.session_state.models is None:
                st.warning("Please upload or place model weights (`forward_model.pt` & `policy.pt`) to initialize.")
                return

            st.markdown("#### TARGET MECHANICAL PROPERTIES")
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
            
            st.markdown(f"""
            <div style="font-family:'Space Mono'; font-size:0.85rem; color:#aaa; margin: 1.5rem 0;">
                Calculated Ac3 Point: <strong style="color:#FFF;">{ac3_est:.1f} °C</strong> 
                (Bounds: {ac3_est+30:.0f}°C – {ac3_est+150:.0f}°C)
            </div>
            """, unsafe_allow_html=True)

            if st.button("SYNTHESIZE OPTIMAL PROCESS (RUN POLICY)"):
                with st.spinner("Executing Deep RL Policy Gradient..."):
                    self.execute_inference(targets)

            if st.session_state.results:
                st.markdown("---")
                st.markdown("#### THEORETICAL PROCESS PRESCRIPTIONS")
                self.render_native_process_cards()

                st.markdown("#### METRIC COMPARISON ANALYSIS")
                fig = self.create_plotly_comparison(targets, st.session_state.results)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # High level detail table
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
                
            st.markdown("#### SIMULATION PARAMETERS")
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

            if st.button("RUN FORWARD SIMULATION"):
                fwd, _, _, _, device = st.session_state.models
                c_arr = np.array([self.comp[e] for e in ELEMENT_NAMES], dtype=np.float32)
                p_arr = np.array([t_aus_f, t_aus_t, q_i, t_tmp_f, t_tmp_t], dtype=np.float32)
                
                with torch.no_grad():
                    p_hat, var = fwd(torch.tensor(c_arr).unsqueeze(0), torch.tensor(p_arr).unsqueeze(0))
                
                pred, std = p_hat[0].numpy(), var[0].sqrt().numpy()
                st.markdown("---")
                
                cols = st.columns(3)
                for i, prop in enumerate(PROPERTY_NAMES):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="background:#0a0a0a; border:1px solid #333; padding:1rem; border-radius:4px; margin-bottom:1rem;">
                            <div style="font-family:'Space Mono'; color:#888; font-size:0.8rem;">{prop}</div>
                            <div style="font-family:'Space Mono'; color:#FFF; font-size:1.5rem; font-weight:bold;">{pred[i]:.1f} <span style="font-size:1rem; color:#aaa;">{PROP_UNITS[prop]}</span></div>
                            <div style="font-family:'Space Mono'; color:#555; font-size:0.75rem;">± {std[i]:.1f} uncertainty</div>
                        </div>
                        """, unsafe_allow_html=True)


    def run(self):
        """Entry point mapping to execute the Application."""
        self.render_sidebar()
        self.render_main()


if __name__ == "__main__":
    app = SteelRecommenderPro()
    app.run()