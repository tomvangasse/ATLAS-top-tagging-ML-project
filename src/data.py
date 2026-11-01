import h5py
import pandas as pd

LABEL = "labels"
WEIGHT = "training_weights"
FEATURES = [
    "fjet_C2",
    "fjet_D2",
    "fjet_ECF1",
    "fjet_ECF2",
    "fjet_ECF3",
    "fjet_L2",
    "fjet_L3",
    "fjet_Qw",
    "fjet_Split12",  
    "fjet_Split23",
    "fjet_Tau1_wta",
    "fjet_Tau2_wta",
    "fjet_Tau3_wta",
    "fjet_Tau4_wta",
    "fjet_ThrustMaj",
]

def load(path, n=None):
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:n] for k in f.keys() if f[k].ndim == 1})
