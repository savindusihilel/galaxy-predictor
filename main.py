from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import torch
import numpy as np
import math
import joblib
import json
import os

from contextlib import asynccontextmanager

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

FEATURE_NAMES = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "g-r",
    "u-g",
    "r-i",
    "Mr",
    "redshift"
]

models = {}


# ======================
# MODEL LOADING
# ======================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading models...")

    models["scaler"] = joblib.load(os.path.join(ASSETS_DIR, "scaler.joblib"))

    with open(os.path.join(ASSETS_DIR, "priors.json"), "r") as f:
        models["priors"] = json.load(f)

    joint = PINNJoint(INPUT_DIM, context_dim=CONTEXT_DIM).to(DEVICE)

    sd = torch.load(
        os.path.join(ASSETS_DIR, "pinn_stageC_joint_final.pth"),
        map_location=DEVICE
    )

    sd = {k: v for k, v in sd.items() if not k.startswith("flow.")}

    joint.load_state_dict(sd, strict=False)
    joint.eval()

    models["joint"] = joint

    flow = build_conditional_maf(
        context_dim=CONTEXT_DIM,
        n_blocks=6,
        hidden_features=64
    ).to(DEVICE)

    flow.load_state_dict(
        torch.load(
            os.path.join(ASSETS_DIR, "pinn_stageC_flow_final.pth"),
            map_location=DEVICE
        )
    )

    flow.eval()

    models["flow"] = flow

    rf_m = os.path.join(ASSETS_DIR, "rf_mass.joblib")
    rf_s = os.path.join(ASSETS_DIR, "rf_sfr.joblib")

    if os.path.exists(rf_m):
        models["rf_mass"] = joblib.load(rf_m)

    if os.path.exists(rf_s):
        models["rf_sfr"] = joblib.load(rf_s)

    # Load demo datasets for validation/test comparison
    demo_path = os.path.join(ASSETS_DIR, "demo_datasets.npz")
    if os.path.exists(demo_path):
        demo_data = np.load(demo_path)
        models["demo"] = {
            "X_val": demo_data["X_val"],
            "yM_val": demo_data["yM_val"],
            "yS_val": demo_data["yS_val"],
            "X_test": demo_data["X_test"],
            "yM_test": demo_data["yM_test"],
            "yS_test": demo_data["yS_test"]
        }
        print("Demo datasets loaded.")

    # Load Random Forest benchmark metrics
    rf_metrics_path = os.path.join(ASSETS_DIR, "rf_metrics.json")
    if os.path.exists(rf_metrics_path):
        with open(rf_metrics_path, "r") as f:
            models["rf_metrics"] = json.load(f)
        print("RF metrics loaded.")

    # Compute PINN metrics from demo validation set
    if "demo" in models and "joint" in models and "scaler" in models:
        try:
            X_val = models["demo"]["X_val"]
            yM_val = models["demo"]["yM_val"]
            yS_val = models["demo"]["yS_val"]

            X_val_scaled = models["scaler"].transform(X_val)
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                out = models["joint"](X_val_t)
            pred_mass = out["mu_mass"].cpu().numpy().flatten()
            pred_sfr = out["mu_sfr"].cpu().numpy().flatten()

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            models["pinn_metrics"] = {
                "mass_metrics": {
                    "rmse": float(np.sqrt(mean_squared_error(yM_val, pred_mass))),
                    "mae": float(mean_absolute_error(yM_val, pred_mass)),
                    "r2": float(r2_score(yM_val, pred_mass))
                },
                "sfr_metrics": {
                    "rmse": float(np.sqrt(mean_squared_error(yS_val, pred_sfr))),
                    "mae": float(mean_absolute_error(yS_val, pred_sfr)),
                    "r2": float(r2_score(yS_val, pred_sfr))
                }
            }
            print("PINN metrics computed.")
        except Exception as e:
            print(f"Failed to compute PINN metrics: {e}")

    # Load SDSS main sequence data
    ms_path = os.path.join(ASSETS_DIR, "main_sequence_data.npz")
    if os.path.exists(ms_path):
        ms_data = np.load(ms_path)
        models["main_sequence"] = {
            "mass": ms_data["mass"].tolist(),
            "sfr": ms_data["sfr"].tolist()
        }
        print(f"Main sequence data loaded ({len(ms_data['mass'])} galaxies).")

    print("Models loaded.")

    yield

    models.clear()


app = FastAPI(
    title="Galaxy Predictor API",
    lifespan=lifespan
)

STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/training-loss")
async def get_training_loss():

    history_path = os.path.join(ASSETS_DIR, "training_history.json")

    print("Looking for:", history_path)
    print("Files in assets:", os.listdir(ASSETS_DIR))

    if not os.path.exists(history_path):
        return {
            "epochsA": [],
            "epochsB": [],
            "epochsC": [],
            "stageA_loss": [],
            "stageB_loss": [],
            "stageC_loss": []
        }

    with open(history_path, "r") as f:
        history = json.load(f)

    stageA = history.get("stageA", {})
    stageB = history.get("stageB", {})
    stageC = history.get("stageC", {})

    return {
        "epochsA": stageA.get("epoch", []),
        "epochsB": stageB.get("epoch", []),
        "epochsC": stageC.get("epoch", []),

        "stageA_loss": stageA.get("train_loss", []),
        "stageB_loss": stageB.get("flow_nll", []),
        "stageC_loss": stageC.get("train_loss", []),

        "total_loss": stageC.get("train_loss", []),
        "physics_loss": stageC.get("train_loss", [])
    }


@app.get("/demo-galaxies")
async def get_demo_galaxies(dataset: str = "val", n: int = 20):
    if "demo" not in models:
        return {"galaxies": []}

    if dataset == "test":
        X = models["demo"]["X_test"]
        yM = models["demo"]["yM_test"]
        yS = models["demo"]["yS_test"]
    else:
        X = models["demo"]["X_val"]
        yM = models["demo"]["yM_val"]
        yS = models["demo"]["yS_val"]

    n = min(n, len(X))
    galaxies = []

    for i in range(n):
        row = X[i]
        galaxies.append({
            "id": int(i),
            "features": row.tolist(),
            "true_mass": float(yM[i]),
            "true_sfr": float(yS[i])
        })

    return {"galaxies": galaxies}


@app.get("/main-sequence")
async def get_main_sequence():
    if "main_sequence" not in models:
        return {"mass": [], "sfr": []}
    return models["main_sequence"]


@app.get("/rf-metrics")
async def get_rf_metrics():
    result = {}
    if "rf_metrics" in models:
        result["rf"] = models["rf_metrics"]
    if "pinn_metrics" in models:
        result["pinn"] = models["pinn_metrics"]
    if not result:
        return {"error": "No metrics loaded"}
    return result


# ======================
# HELPER FUNCTIONS
# ======================

def compute_absolute_magnitude(r_mag, redshift):
    """
    Approximate absolute magnitude M_r
    """
    c = 3e5  # km/s
    H0 = 70  # km/s/Mpc

    d_mpc = (c / H0) * redshift
    d_pc = d_mpc * 1e6

    if d_pc <= 0:
        return -20

    return r_mag - 5 * math.log10(d_pc) + 5

def compute_saliency(model, x_scaled):

    x_t = torch.tensor(x_scaled, dtype=torch.float32, requires_grad=True)

    out = model(x_t)

    m = out["mu_mass"]
    s = out["mu_sfr"]

    model.zero_grad()

    m.backward(retain_graph=True)
    grad_mass = x_t.grad.detach().cpu().numpy()[0]

    x_t.grad.zero_()

    s.backward()
    grad_sfr = x_t.grad.detach().cpu().numpy()[0]

    grad_mass = np.abs(grad_mass)
    grad_sfr = np.abs(grad_sfr)

    grad_mass = grad_mass / (grad_mass.sum() + 1e-8)
    grad_sfr = grad_sfr / (grad_sfr.sum() + 1e-8)

    mass_importance = dict(zip(FEATURE_NAMES, grad_mass.tolist()))
    sfr_importance = dict(zip(FEATURE_NAMES, grad_sfr.tolist()))

    return mass_importance, sfr_importance


def predict_quenching_probability_logic(joint, flow, x_t, n_samples=256):

    with torch.no_grad():

        ctx = joint(x_t)["context"]

        q = flow.sample(n_samples, context=ctx)

        q = q.squeeze().cpu().numpy()

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

    quenching_posterior: list[float]

    mass_feature_importance: dict
    sfr_feature_importance: dict

    rf_mass_log_mean: float | None = None
    rf_mass_log_std: float | None = None

    rf_sfr_log_mean: float | None = None
    rf_sfr_log_std: float | None = None


# ======================
# PREDICTION ENDPOINT
# ======================

@app.post("/predict", response_model=PredictionResult)
async def predict(data: GalaxyInput):

    Mr = compute_absolute_magnitude(data.r, data.redshift)

    features = [
        data.u,
        data.g,
        data.r,
        data.i,
        data.z,
        data.g - data.r,
        data.u - data.g,
        data.r - data.i,
        Mr,
        data.redshift
    ]

    x = np.array([features], dtype=np.float32)

    x_scaled = models["scaler"].transform(x)

    x_t = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():

        out = models["joint"](x_t)

    m_mu = out["mu_mass"].item()
    s_mu = out["mu_sfr"].item()

    sigma_m = float(np.sqrt(np.exp(out["logvar_mass"].item())))
    sigma_s = float(np.sqrt(np.exp(out["logvar_sfr"].item())))

    mass_imp, sfr_imp = compute_saliency(models["joint"], x_scaled)

    q_mean, q_std, q_samples = predict_quenching_probability_logic(
        models["joint"],
        models["flow"],
        x_t,
        n_samples=512
    )

    q_std = min(q_std, 0.25)

    rf_res = {}

    if "rf_mass" in models and "rf_sfr" in models:

        rf_m, rf_m_std = rf_predict_with_uncertainty(
            models["rf_mass"],
            x_scaled
        )

        rf_s, rf_s_std = rf_predict_with_uncertainty(
            models["rf_sfr"],
            x_scaled
        )

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

        mass_feature_importance=mass_imp,
        sfr_feature_importance=sfr_imp,

        **rf_res
    )