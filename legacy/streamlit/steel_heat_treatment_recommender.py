"""MAST: a focused materials-design research workspace."""
import html
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from mast.catalog import ELEMENT_NAMES, PROPERTY_NAMES, QUENCH_MAP, PROP_UNITS, COMP_RANGES, PROP_RANGES, STEEL_PRESETS
from mast.engine import FAMILIES, Constraints, load_models, recommend, validate_composition, candidate_mask
from mast.models import compute_ac3_np

ROOT = Path(__file__).resolve().parent
LABELS = {'UTS':'Tensile strength', 'YS':'Yield strength', 'elongation':'Elongation',
          'reduction_of_area':'Reduction of area', 'HRC':'Hardness', 'impact_energy_J':'Impact energy'}
UNITS = {**PROP_UNITS, 'HRC':'HRC'}
st.set_page_config(page_title='MAST · Materials workspace', page_icon='◈', layout='wide')
st.markdown((ROOT/'assets/workspace.css').read_text(encoding='utf-8'), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def bootstrap_models(forward=None, policy=None):
    if forward and policy:
        return load_models(forward, policy)
    f, p = ROOT/'checkpoints/forward_model.pt', ROOT/'checkpoints/policy.pt'
    return load_models(f, p) if f.exists() and p.exists() else None


def initialize():
    if 'models' not in st.session_state:
        try:
            st.session_state.models = bootstrap_models()
        except Exception as exc:
            st.session_state.models = None
            st.session_state.model_error = str(exc)
    st.session_state.setdefault('results', None)


def section(number, title, description=None):
    st.markdown(f'<div class="section-heading"><span>{number}</span><h3>{title}</h3></div>', unsafe_allow_html=True)
    if description:
        st.caption(description)


def evidence_note():
    with st.expander('Research model · Not validated for manufacturing'):
        st.write('This model was trained on synthetic data. A phase-transformation defect was identified in its training sources; predictions are not validated for manufacturing.')
        st.write('Ensemble spread is not calibrated confidence. Only the nominal 25 mm steel scenario is supported. Hold time excludes heating ramps, transfer, quenching and cooling.')
        st.caption('Use results to compare research candidates and plan independently reviewed experiments.')


def material_setup():
    with st.container(border=True):
        section('01', 'Choose your material')
        a, b = st.columns(2)
        family = a.selectbox('Material family', list(FAMILIES), key='family')
        if family != 'Low-alloy steel':
            b.markdown('**Model in development**')
            st.info(FAMILIES[family][1])
            return None
        preset = b.selectbox('Alloy preset', [p for p in STEEL_PRESETS if p != 'H13 Tool'], key='preset')
        values = STEEL_PRESETS[preset]
        comp = {}
        with st.expander('Composition · inspect or adjust weight percentages', expanded=preset == 'Custom'):
            st.caption('Illustrative compositions, not certified grade specifications. H13 is outside this model’s supported range.')
            cols = st.columns(3)
            for i, e in enumerate(ELEMENT_NAMES[:-1]):
                lo, hi = COMP_RANGES[e]
                default = float(values[e]) if values else (lo+hi)/2
                comp[e] = cols[i % 3].number_input(f'{e} (wt%)', float(lo), float(hi), default,
                    step=.0001 if e == 'B' else .001, format='%.4f' if e == 'B' else '%.3f', key=f'composition_{preset}_{e}')
            comp['Fe'] = 100-sum(comp.values())
            st.caption(f"Iron balance: {comp['Fe']:.3f} wt% · Nominal section diameter: 25 mm")
        st.markdown(f'<div class="material-strip"><strong>{html.escape(preset)}</strong><span>C {comp["C"]:.2f}%</span><span>Cr {comp["Cr"]:.2f}%</span><span>Ni {comp["Ni"]:.2f}%</span><span>25 mm section</span></div>', unsafe_allow_html=True)
        return comp


def ac3_value(comp):
    return compute_ac3_np(comp['C'], comp['Mn'], comp['Si'], comp['Ni'], comp['Cr'], comp['Mo'], comp['V'], comp['Cu'])


def search_settings(include_search=True):
    with st.expander('Equipment & search settings' if include_search else 'Equipment limits'):
        st.caption('Limit candidates to the equipment and processing time available to you.')
        a, b = st.columns(2)
        temperature = a.number_input('Furnace limit (°C)', 750., 1100., 1100.)
        hold = b.number_input('Combined hold limit (h)', 1., 12., 12.)
        quench = st.multiselect('Available quench media', list(QUENCH_MAP.values()), default=list(QUENCH_MAP.values()))
        if not include_search:
            return Constraints(temperature, hold, tuple(q for q, label in QUENCH_MAP.items() if label in quench))
        tolerance = st.slider('Target tolerance (%)', 1, 50, 15, help='Every property must fall within this percentage of its target to count as a match.')
        time_weight = st.slider('Preference for shorter holds', 0., 2., 0., .1)
        st.divider()
        a, b = st.columns(2)
        method = a.selectbox('Search method', ['hybrid', 'random', 'policy'])
        seed = b.number_input('Sampling seed', 0, 2147483647, 42)
        budget = a.select_slider('Candidates to evaluate', options=[100, 200, 300, 500, 750, 1000], value=300)
        top_k = b.number_input('Results to display', 1, 10, 4)
        st.caption('Hybrid combines the learned policy with random search. A seed lets you repeat the same search.')
    constraints = Constraints(temperature, hold, tuple(q for q, label in QUENCH_MAP.items() if label in quench))
    return dict(n_candidates=budget, top_k=int(top_k), seed=int(seed), constraints=constraints,
                method=method, tolerance_fraction=tolerance/100, time_weight=time_weight)


def fingerprint(comp, targets, settings):
    return json.dumps(dict(composition=comp, targets=targets, **{**settings, 'constraints':asdict(settings['constraints'])}), sort_keys=True)


def comparison(result, targets, tolerance):
    names = [LABELS[p] for p in PROPERTY_NAMES]
    values = [100*result['pred'][p]/targets[p] for p in PROPERTY_NAMES]
    spread = [100*result['std'][p]/targets[p] for p in PROPERTY_NAMES]
    fig = go.Figure(go.Scatter(x=values, y=names, mode='markers',
        marker=dict(color='#24594f', size=9), error_x=dict(type='data', array=spread, thickness=1.2, color='#8ba39b'),
        hovertemplate='%{y}: %{x:.1f}% of target<extra></extra>'))
    fig.add_vrect(x0=100*(1-tolerance), x1=100*(1+tolerance), fillcolor='#edf3ef', line_width=0, layer='below')
    fig.add_vline(x=100, line_dash='dot', line_color='#68756f', line_width=1)
    fig.update_layout(height=285, margin=dict(l=0,r=15,t=15,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif', color='#4e5a55',size=12), showlegend=False,
        xaxis=dict(title='Percent of target', ticksuffix='%', gridcolor='#ebeeeb', zeroline=False),
        yaxis=dict(autorange='reversed', showgrid=False))
    return fig


def show_results(report):
    results = report['results']
    st.markdown('<div class="eyebrow">SEARCH RESULTS</div>', unsafe_allow_html=True)
    if not results:
        st.subheader('No candidates passed screening')
        st.write('Review your equipment limits and try another search. This does not prove that your target is impossible.')
    else:
        matches = sum(r['within_tolerance'] for r in results)
        st.subheader(f'{len(results)} candidates to explore')
        st.caption(f"{report['valid_candidates']} of {report['candidate_budget']} evaluated candidates passed screening · seed {report['seed']}")
        if matches:
            st.success(f'{matches} displayed candidate(s) meet all target tolerances in the model.')
        else:
            st.markdown('<div class="result-notice"><strong>No complete target match.</strong> These are the closest screened candidates in this search.</div>', unsafe_allow_html=True)
        labels = [f"Candidate {r['rank']} · {r['quench']} · {r['hold_hours']:.1f} h hold" for r in results]
        chosen = st.selectbox('Inspect a candidate', labels, key='selected_candidate')
        r = results[labels.index(chosen)]
        a, b, c = st.columns(3)
        a.metric('Austenitize', f"{r['T_aus']:.0f} °C")
        a.caption(f"{r['t_aus']:.2f} h soak")
        b.metric('Quench', r['quench'])
        c.metric('Temper', f"{r['T_temper']:.0f} °C")
        c.caption(f"{r['t_temper']:.2f} h hold")
        st.markdown('#### How close is the prediction?')
        st.plotly_chart(comparison(r, report['targets'], report['tolerance_fraction']), width='stretch', config={'displayModeBar':False})
        st.caption('Shaded band: target tolerance. Dots: predictions. Whiskers: ensemble spread, not a confidence interval.')
        with st.expander('Property values & candidate details'):
            rows = [{'Property':LABELS[p], 'Unit':UNITS[p], 'Target':report['targets'][p],
                     'Prediction':r['pred'][p], 'Ensemble spread':r['std'][p]} for p in PROPERTY_NAMES]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', column_config={
                'Target':st.column_config.NumberColumn(format='%.2f'), 'Prediction':st.column_config.NumberColumn(format='%.2f'),
                'Ensemble spread':st.column_config.NumberColumn(format='%.2f')})
            st.caption(f"Ranking score: {r['score']:.3f} · Generator: {r['generator']} · Pareto candidate: {'yes' if r['pareto'] else 'no'}")
    st.download_button('Export research run', json.dumps(report, indent=2, allow_nan=False), 'mast-research-run.json', 'application/json', width='stretch')


def empty_results():
    st.markdown("""<div class="empty-state"><div class="empty-symbol" aria-hidden="true">◎</div>
<div class="eyebrow">YOUR NEXT EXPERIMENT STARTS HERE</div><h2>A clear target.<br>A considered process.</h2>
<p>Set the properties you want to achieve. Compare screened heat-treatment candidates here.</p>
<div class="process-path"><span>Material</span><i>→</i><span>Targets</span><i>→</i><span>Process</span></div>
<div class="empty-foot">Predictions for research. Validation for confidence.</div></div>""", unsafe_allow_html=True)


def design(comp):
    with st.container(key='workspace'):
        left, right = st.columns([1, 1.15], gap='large')
        with left, st.container(key='design_panel'):
            section('02', 'Define your targets', 'Enter the properties you want the heat treatment to achieve.')
            targets = {}
            for pair in (('UTS','YS'), ('elongation','reduction_of_area'), ('HRC','impact_energy_J')):
                cols = st.columns(2)
                for col, prop in zip(cols, pair):
                    lo, hi, default = PROP_RANGES[prop]
                    targets[prop] = col.number_input(f'{LABELS[prop]} ({UNITS[prop]})', float(lo), float(hi), float(default), step=1., key=f'target_{prop}')
            settings = search_settings()
            st.caption(f"Point targets · ±{settings['tolerance_fraction']*100:.0f}% tolerance · {settings['n_candidates']} candidates")
            request_id = fingerprint(comp, targets, settings)
            if st.session_state.get('request_id') != request_id:
                st.session_state.results = None
                st.session_state.pop('report', None)
            clicked = st.button('Find candidate processes', type='primary', width='stretch', disabled=st.session_state.models is None)
            if clicked:
                try:
                    with st.spinner('Screening candidate processes…'):
                        report = recommend(st.session_state.models, comp, targets, **settings)
                    st.session_state.report = report
                    st.session_state.results = report['results']
                    st.session_state.request_id = request_id
                    st.session_state.pop('selected_candidate', None)
                except ValueError as exc:
                    st.error(f'Check your design inputs: {exc}')
            st.caption('Combined hold time is not the complete manufacturing cycle.')
        with right, st.container(border=True, key='results_panel'):
            if 'report' in st.session_state:
                show_results(st.session_state.report)
            else:
                empty_results()


def forward(comp):
    a, b = st.columns([1, 1.15], gap='large')
    with a, st.container(key='design_panel'):
        section('02', 'Set the heat treatment')
        x, y = st.columns(2)
        aus = x.number_input('Austenitize temperature (°C)',750.,1100.,870.)
        soak = y.number_input('Austenitize hold (h)',.5,4.,1.)
        quench = st.selectbox('Quench medium',list(QUENCH_MAP.values()))
        x, y = st.columns(2)
        temper = x.number_input('Temper temperature (°C)',150.,700.,450.)
        hold = y.number_input('Temper hold (h)',.5,8.,2.)
        constraints = search_settings(include_search=False)
        clicked = st.button('Predict properties',type='primary',width='stretch', disabled=st.session_state.models is None)
    with b, st.container(border=True):
        if not clicked:
            st.subheader('Understand a process')
            st.write('Enter a heat treatment to estimate its properties. Predictions appear here with ensemble spread.')
        else:
            try:
                validate_composition(comp)
                q = next(k for k,v in QUENCH_MAP.items() if v == quench)
                process = np.array([[aus, soak, q, temper, hold]],dtype=np.float32)
                composition = torch.tensor([[comp[e] for e in ELEMENT_NAMES]],dtype=torch.float32)
                with torch.inference_mode():
                    mean, variance = st.session_state.models[0](composition,torch.tensor(process))
                if not candidate_mask(process,mean.numpy(),variance.numpy(),ac3_value(comp),constraints)[0]:
                    raise ValueError('The process or its predictions fall outside supported physical limits.')
                st.subheader('Predicted properties')
                for i,p in enumerate(PROPERTY_NAMES):
                    st.markdown(f'**{LABELS[p]}**')
                    st.write(f'{mean[0,i].item():.1f} {UNITS[p]}  ·  ± {variance[0,i].sqrt().item():.2f} ensemble spread')
                st.caption('Research estimates only. Ensemble spread is not calibrated confidence.')
            except ValueError as exc:
                st.error(str(exc))


def library():
    st.subheader('One workspace. Distinct material models.')
    st.caption('Availability reflects implemented capability, not a claim of industrial qualification.')
    for family, (status, detail) in FAMILIES.items():
        with st.container(border=True):
            st.markdown(f'**{family}**  ·  {"Research model" if status == "research" else "In development"}')
            st.write(detail)


def main():
    initialize()
    st.markdown('<div class="brandbar"><div class="brand"><span class="brand-mark">M</span> MAST <span class="brand-divider">/</span><span class="brand-context">Materials workspace</span></div><span class="research-badge">RESEARCH PREVIEW</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading"><div class="eyebrow">FROM PROPERTIES TO PROCESS</div><h1>Design with a target in mind.</h1><p>Explore material performance. Compare the processes that could get you there.</p></div>', unsafe_allow_html=True)
    mode = st.radio('Workspace', ['Design a process', 'Predict properties', 'Material library'], horizontal=True, label_visibility='collapsed', key='workspace_mode')
    evidence_note()
    if st.session_state.models is None:
        st.info('Connect compatible model files to enable prediction.')
        with st.expander('Connect model files', expanded=True):
            f = st.file_uploader('Forward model (.pt)', type=['pt'])
            p = st.file_uploader('Policy model (.pt)', type=['pt'])
            if f and p:
                try:
                    st.session_state.models = bootstrap_models(f.getvalue(),p.getvalue())
                    st.rerun()
                except Exception as exc:
                    st.error(f'Model files could not be loaded: {exc}')
    if mode == 'Material library':
        library()
    else:
        comp = material_setup()
        if comp is not None:
            st.write('')
            if mode == 'Design a process':
                design(comp)
            else:
                forward(comp)
    st.markdown('<div class="workspace-footer"><span>MAST · Materials inverse design</span><span>Evidence before application.</span></div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
