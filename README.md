# Interpretable Galaxy Property Predictor

**Physics-Informed Neural Networks with Explainable Uncertainty**

This project predicts stellar mass, star formation rate (SFR), and quenching probability from galaxy photometry using a PINN (Physics-Informed Neural Network) and Normalizing Flows. It provides physically meaningful uncertainty estimates.

![Galaxy Predictor UI](https://via.placeholder.com/800x400?text=Galaxy+Predictor+App)

## Features
- **Modern Web App**: Built with FastAPI and Vanilla JS (Cosmic Glass theme).
- **Physical Uncertainty**: Distinguishes between statistical noise and physical degeneracy.
- **Interactive Interface**: Use presets or enters custom photometry values.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the web server:
```bash
uvicorn main:app --reload
```

Open your browser at **http://127.0.0.1:8000**.

## Project Structure
- `main.py`: FastAPI entry point.
- `static/`: HTML/CSS/JS frontend.
- `assets/`: Model weights and scalers.
- `data/`: Raw datasets (ignored in git).
- `notebooks/`: Research notebooks.
- `legacy/`: Old POC scripts.

## Author
Savindu Sihilel  
SLIIT