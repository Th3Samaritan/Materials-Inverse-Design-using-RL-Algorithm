# MAST Inverse Design — Streamlit App

Steel heat treatment inverse design powered by the MAST policy gradient model.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. **Upload models** — drag `forward_model.pt` and `policy.pt` into the sidebar file uploaders
2. **Set composition** — use the element sliders or pick a preset (4340, 1045, 4140, H13)
3. **Set targets** — enter the mechanical properties you want to achieve
4. **Recommend** — click "Get Heat Treatment Recommendations"

The app returns the top-N ranked heat treatment processes, each with:
- Austenitizing temperature and time
- Quench medium
- Tempering temperature and time
- Predicted properties with uncertainty (±1 std from ensemble)
- Score (higher = better property match)

## Two Modes

**Inverse Design (main):** Composition + Target Properties → Recommended Heat Treatment
**Forward Prediction:** Composition + Heat Treatment → Predicted Properties

## Model Files

Download from your Kaggle run Output tab:
- `checkpoints/forward_model.pt`
- `checkpoints/policy.pt`

Or extract from `MAST_model_bundle.zip`.

## Physical Limits

Some targets are physically unreachable for a given composition — for example,
a 4340-like steel cannot achieve UTS < ~950 MPa with standard quench-and-temper
even at maximum tempering. The app shows a warning when the top recommendation
overshoots the target by >15%. In those cases, use a lower-hardenability steel.

## Deployment

Deploy to Streamlit Community Cloud:
1. Push this folder to a GitHub repo
2. Go to share.streamlit.io
3. Connect repo, set main file as `app.py`
4. Upload model files at runtime via the sidebar uploaders

For local deployment with pre-loaded models, replace the `st.file_uploader`
calls with `torch.load('path/to/model.pt')` directly.
