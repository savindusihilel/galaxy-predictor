from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import json
import os
from contextlib import asynccontextmanager

# Relative imports assuming running as a module (e.g. uvicorn api.main:app)
from models import PINNJoint
from flow_utils import build_conditional_maf

# ======================
# CONFIGURATION
# ======================
DEVICE = "cpu"
INPUT_DIM = 10
CONTEXT_DIM = 64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Global state for models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    print("Loading models...")
    try:
        models["scaler"] = joblib.load(os.path.join(ASSETS_DIR, "scaler.joblib"))
        
        with open(os.path.join(ASSETS_DIR, "priors.json"), "r") as f:
            models["priors"] = json.load(f)

        joint = PINNJoint(INPUT_DIM, context_dim=CONTEXT_DIM).to(DEVICE)
        # Load weights, filtering out flow params if any mixed in (as per original app.py logic)
        sd = torch.load(os.path.join(ASSETS_DIR, "pinn_stageC_joint_final.pth"), map_location=DEVICE)
        sd = {k: v for k, v in sd.items() if not k.startswith("flow.")}
        joint.load_state_dict(sd, strict=False)
        joint.eval()
        models["joint"] = joint

        flow = build_conditional_maf(
            context_dim=CONTEXT_DIM,
            n_blocks=4,
            hidden_features=64
        ).to(DEVICE)
        flow.load_state_dict(
            torch.load(os.path.join(ASSETS_DIR, "pinn_stageC_flow_final.pth"), map_location=DEVICE)
        )
        flow.eval()
        models["flow"] = flow
        
        # Optional RF models
        rf_m_path = os.path.join(ASSETS_DIR, "rf_mass.joblib")
        rf_s_path = os.path.join(ASSETS_DIR, "rf_sfr.joblib")
        if os.path.exists(rf_m_path):
            models["rf_mass"] = joblib.load(rf_m_path)
        if os.path.exists(rf_s_path):
            models["rf_sfr"] = joblib.load(rf_s_path)
            
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e
    
    yield
    
    # Clean up if needed
    models.clear()

app = FastAPI(title="Galaxy Predictor API", lifespan=lifespan)

# Mount static files
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
def predict_quenching_probability_logic(joint, flow, x_t, n_samples=256):
    with torch.no_grad():
        ctx = joint(x_t)["context"]
        q = flow.sample(n_samples, context=ctx).squeeze().cpu().numpy()
        q = np.clip(q, 0, 1)
    return float(q.mean()), float(q.std()), q.tolist()

def rf_predict_with_uncertainty(rf_model, x):
    preds = np.array([tree.predict(x) for tree in rf_model.estimators_])
    return float(preds.mean()), float(preds.std())

# ======================
# DATA MODELS
# ======================
class GalaxyInput(BaseModel):
    u: float
    g: float
    r: float
    i: float
    z: float
    redshift: float

class PredictionResult(BaseModel):
    mass_log_mean: float
    mass_log_std: float
    sfr_log_mean: float
    sfr_log_std: float
    quenching_prob_mean: float
    quenching_prob_std: float
    quenching_posterior: list[float]  # Histogram samples
    
    # Random Forest comparison (optional/nullable)
    rf_mass_log_mean: float | None = None
    rf_mass_log_std: float | None = None
    rf_sfr_log_mean: float | None = None
    rf_sfr_log_std: float | None = None

# ======================
# ENDPOINTS
# ======================
@app.post("/predict", response_model=PredictionResult)
async def predict(data: GalaxyInput):
    # Prepare input
    # Original order: u, g, r, i, z, g-r, u-g, r-i, 0.0, redshift
    # Note: 0.0 is a placeholder column likely used in training, preserved here.
    
    features = [
        data.u,
        data.g,
        data.r,
        data.i,
        data.z,
        data.g - data.r,
        data.u - data.g,
        data.r - data.i,
        0.0,
        data.redshift
    ]
    
    x = np.array([features], dtype=np.float32)
    x_scaled = models["scaler"].transform(x)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)
    
    # PINN Prediction
    with torch.no_grad():
        out = models["joint"](x_t)
        
    m_mu = out["mu_mass"].item()
    s_mu = out["mu_sfr"].item()
    sigma_m = float(np.sqrt(np.exp(out["logvar_mass"].item())))
    sigma_s = float(np.sqrt(np.exp(out["logvar_sfr"].item())))
    
    q_mean, q_std, q_samples = predict_quenching_probability_logic(
        models["joint"],
        models["flow"],
        x_t,
        n_samples=512 
    )
    
    # Heuristic clip from original app
    q_std = min(q_std, 0.25)
    
    # RF Comparison
    rf_res = {}
    if "rf_mass" in models and "rf_sfr" in models:
        rf_m, rf_m_std = rf_predict_with_uncertainty(models["rf_mass"], x_scaled)
        rf_s, rf_s_std = rf_predict_with_uncertainty(models["rf_sfr"], x_scaled)
        rf_res["rf_mass_log_mean"] = rf_m
        rf_res["rf_mass_log_std"] = rf_m_std
        rf_res["rf_sfr_log_mean"] = rf_s
        rf_res["rf_sfr_log_std"] = rf_s_std

    return PredictionResult(
        mass_log_mean=m_mu,
        mass_log_std=sigma_m,
        sfr_log_mean=s_mu,
        sfr_log_std=sigma_s,
        quenching_prob_mean=q_mean,
        quenching_prob_std=q_std,
        quenching_posterior=q_samples,
        **rf_res
    )

# Removed simple root endpoint in favor of static file serving
