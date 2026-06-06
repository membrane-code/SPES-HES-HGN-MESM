import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
import pickle
import json
import threading
import queue
from functools import lru_cache
from tkinter import filedialog, messagebox
from typing import Dict, Tuple, Any, Optional

# Try importing scipy for optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================
# ------------------- 0. GLOBAL STYLES -----------------------
# ============================================================

# Initial Appearance Mode (Fixed to Light)
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green") 

# --- COLOR PALETTE (Tuple format: "Light Color", "Dark Color") ---
# Kept in tuple format for CTk compatibility, but we will force Light [0] usage for plots

COLORS = {
    "bg_main":       ("#F3F4F6", "#1a1a1a"),  # Light Grey
    "bg_card":       ("#F9FAFB", "#2b2b2b"),  # Slightly Lighter Grey
    "text_main":     ("black", "white"),      # Black
    "text_sub":      ("#6B7280", "#a3a3a3"),  # Dark Grey
    "primary":       ("#6FE0B3", "#6FE0B3"),  # Green
    "primary_dark":  ("#5BC299", "#5BC299"),  # Hover Green
    "secondary":     ("#8C8EDF", "#8C8EDF"),  # Blue
    "secondary_hov": ("#7CA0D0", "#7CA0D0"),
    "accent":        ("#6F30A2", "#D8B4E2"),  # Deep Violet
    "destructive":   ("#CC0098", "#CC0098"),  # Red/Pink
    "destructive_h": ("#A00078", "#A00078"),
    "chart_bg":      ("#F9FAFB", "#2b2b2b"),  # Matches Card
    "chart_lines":   ("#E5E5E5", "#444444"),  # Grid lines
    "input_bg":      ("white", "#333333"),
    "warn_light":    ("#FEE2E2", "#500000"),  # Extrapolation warning bg
}

# Set Global Font for Matplotlib (Default Light)
plt.rcParams['font.family'] = 'Cambria'
plt.rcParams['font.size'] = 8
plt.rcParams['mathtext.fontset'] = 'cm'

# ============================================================
# ------------------- 1. SCIENTIFIC CORE ---------------------
# ============================================================

class ScientificEngine:
    def __init__(self):
        self.inputs = ["Pressure", "Conc", "pH"]
        self.outputs = ["Flux", "SMX", "TRM", "TET", "ERY"]
        
        # Exact Fallback Weights
        self.meta_params = {
            "Flux": [-0.068333285, 1.373323147, -0.104341967, -0.54489965],
            "SMX":  [0.01556722,   1.335469746,  0.174309074, -0.720833601],
            "TRM":  [0.047666143, -0.580855515,  1.129398511,  0.317550521],
            "TET":  [-0.020643431, 0.414042676,  0.23580088,   0.461972832],
            "ERY":  [-0.005308616, 1.083363941,  0.363772469, -0.679546371]
        }

        self.domain = {
            "Pressure": (0.5, 3.5),
            "Conc": (2.0, 10.0),
            "pH": (5.0, 9.0)
        }

        self.models = {}
        self.experimental_data = None
        self._init_stats()
        self._load_models()

    def _init_stats(self):
        # Hardcoded Experimental Data
        data = [
            [0.5, 2, 7, 19.53, 25.07, 28.04, 25.32, 19.82],
            [0.5, 6, 5, 15.46, 21.81, 29.66, 26.64, 17.52],
            [0.5, 6, 9, 13.17, 18.44, 30.11, 29.97, 9.12],
            [0.5, 10, 7, 17.64, 20.34, 27.45, 27.71, 24.46],
            [2, 2, 5, 60.42, 69.88, 85.05, 80.32, 55.08],
            [2, 2, 9, 53.33, 73.76, 17.72, 61.54, 13.34],
            [2, 6, 7, 58.68, 72.87, 77.22, 72.73, 53.97],
            [2, 10, 5, 52.79, 65.63, 87.31, 85.61, 61.32],
            [2, 10, 9, 49.62, 61.24, 18.08, 67.85, 19.14],
            [3.5, 2, 7, 83.25, 92.41, 91.07, 93.68, 71.65],
            [3.5, 6, 5, 61.74, 71.82, 95.81, 94.15, 70.01],
            [3.5, 6, 9, 50.33, 61.52, 90.67, 99.01, 26.92],
            [3.5, 10, 7, 78.13, 83.58, 92.33, 96.98, 82.13]
        ]
        columns = ["Pressure", "Conc", "pH", "Flux", "SMX", "TRM", "TET", "ERY"]
        self.experimental_data = pd.DataFrame(data, columns=columns)

        # Normalization Stats (Must match trained models)
        self.X_mean = pd.Series([2.0, 6.0, 7.0], index=self.inputs)
        self.X_std = pd.Series([1.1767, 3.1379, 1.5689], index=self.inputs)
        self.Y_mean = pd.Series([47.2377, 56.7977, 59.2708, 66.2700, 40.3446], index=self.outputs)
        self.Y_std = pd.Series([22.6403, 24.9302, 32.0292, 28.0572, 24.7720], index=self.outputs)

    def _load_models(self):
        activations = ["relu", "tanh", "logistic"]
        pkl_files = {act: f"{act}_model.pkl" for act in activations}
        meta_path = "meta_layer_model.pkl"
        
        missing = [f for f in pkl_files.values() if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Missing model file(s):\n" + "\n".join(missing))

        try:
            for act, fname in pkl_files.items():
                with open(fname, 'rb') as f:
                    self.models[act] = pickle.load(f)
            
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta_res = pickle.load(f)
                    W = np.asarray(meta_res["weights"])
                    b = np.asarray(meta_res["intercepts"])
                    for i, name in enumerate(self.outputs):
                        self.meta_params[name] = [b[i]] + list(W[i, :])
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

    def is_extrapolating(self, p: float, c: float, ph: float) -> bool:
        vals = {"Pressure": p, "Conc": c, "pH": ph}
        for k, (min_v, max_v) in self.domain.items():
            if not (min_v <= vals[k] <= max_v):
                return True
        return False

    def predict(self, p: float, c: float, ph: float, model_mode: str = "Ensemble") -> Dict[str, float]:
        X_flat = np.array([[p, c, ph]])
        
        if model_mode == "Ensemble":
            pred_raw, std_raw = self.batch_predict_ensemble(X_flat)
            pred_row, std_row = pred_raw[0], std_raw[0]
        else:
            x_z = ((X_flat - self.X_mean.values) / self.X_std.values)
            key = model_mode.lower()
            mdl = self.models.get(key)
            if mdl is None: raise RuntimeError(f"Model '{key}' not loaded.")
            pz = mdl.predict(x_z)
            if pz.ndim == 1: pz = pz.reshape(-1, 1)
            pred_row = (pz * self.Y_std.values + self.Y_mean.values).flatten()
            std_row = np.zeros_like(pred_row)

        out_dict = {}
        for i, name in enumerate(self.outputs):
            val = pred_row[i]
            if name == "Flux": val = max(0.0, val)
            else: val = max(0.0, min(100.0, val))
            out_dict[name] = float(val)
            out_dict[f"{name}_std"] = float(std_row[i])
            
        out_dict["extrapolating"] = self.is_extrapolating(p, c, ph)
        return out_dict

    def batch_predict_ensemble(self, X_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X_flat, dtype=float)
        mean_X = self.X_mean.values.reshape(1, -1)
        std_X = self.X_std.values.reshape(1, -1)
        X_z = (X - mean_X) / std_X 

        activations = ["relu", "tanh", "logistic"]
        base_preds = []
        for act in activations:
            mdl = self.models[act]
            pz = np.asarray(mdl.predict(X_z))
            if pz.ndim == 1: pz = pz.reshape(-1, 1)
            base_preds.append(pz)
        
        base_preds = np.stack(base_preds, axis=0) 
        n_out = len(self.outputs)
        
        intercepts = np.zeros(n_out, dtype=float)
        weights = np.zeros((n_out, len(activations)), dtype=float)
        
        for i, name in enumerate(self.outputs):
            p = self.meta_params[name]
            intercepts[i] = float(p[0])
            weights[i, :] = np.array(p[1:], dtype=float)

        weighted = np.einsum('a n o, o a -> n o', base_preds, weights)
        avg_z = weighted + intercepts.reshape(1, -1)

        w_abs = np.abs(weights)
        w_sum = w_abs.sum(axis=1, keepdims=True)
        w_sum_safe = np.where(w_sum == 0.0, 1.0, w_sum)
        nw = (w_abs / w_sum_safe)
        
        mean_by_weight = np.einsum('a n o, o a -> n o', base_preds, nw)
        diffs = base_preds - mean_by_weight[None, :, :]
        var_proxy = np.einsum('a n o, o a -> n o', diffs**2, nw)
        std_z = np.sqrt(np.maximum(var_proxy, 0.0))

        Y_mean = self.Y_mean.values.reshape(1, -1)
        Y_std = self.Y_std.values.reshape(1, -1)
        
        pred_raw = avg_z * Y_std + Y_mean
        std_raw = std_z * Y_std

        return pred_raw, std_raw

    def find_optimum(self, constraints: Dict[str, float], weights: Dict[str, float]) -> Optional[Dict[str, float]]:
        if SCIPY_AVAILABLE:
            return self._find_optimum_scipy(constraints, weights)
        else:
            return self._find_optimum_grid(constraints, weights)

    def _find_optimum_scipy(self, constraints: Dict[str, float], weights: Dict[str, float]) -> Optional[Dict[str, float]]:
        def objective(x):
            preds, _ = self.batch_predict_ensemble(x.reshape(1, -1))
            flux = max(0.0, preds[0, 0])
            rejections = np.clip(preds[0, 1:], 0.0, 100.0)
            score = 0
            if weights.get("Flux"):
                z_flux = (flux - self.Y_mean["Flux"]) / self.Y_std["Flux"]
                score += weights["Flux"] * z_flux
            names = ["SMX", "TRM", "TET", "ERY"]
            for i, name in enumerate(names):
                if weights.get(name):
                    z_rej = (rejections[i] - self.Y_mean[name]) / self.Y_std[name]
                    score += weights[name] * z_rej
            return -score

        min_flux = constraints.get('min_flux', 0.0)
        min_rej = constraints.get('min_rej', 0.0)
        cons = []
        cons.append({'type': 'ineq', 'fun': lambda x: self.predict(x[0], x[1], x[2])['Flux'] - min_flux})
        for r_key in ["SMX", "TRM", "TET", "ERY"]:
            cons.append({'type': 'ineq', 'fun': lambda x, k=r_key: self.predict(x[0], x[1], x[2])[k] - min_rej})

        bnds = (self.domain["Pressure"], self.domain["Conc"], self.domain["pH"])
        x0 = [2.0, 6.0, 7.0] 
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-4)
            if res.success:
                final_res = self.predict(res.x[0], res.x[1], res.x[2])
                final_res["Pressure"] = res.x[0]
                final_res["Conc"] = res.x[1]
                final_res["pH"] = res.x[2]
                rej_vals = [final_res[k] for k in ["SMX", "TRM", "TET", "ERY"]]
                final_res['Avg_Rej'] = sum(rej_vals) / len(rej_vals)
                final_res['Rejections'] = {k: final_res[k] for k in ["SMX", "TRM", "TET", "ERY"]}
                return final_res
        except Exception: pass
        return None

    def _find_optimum_grid(self, constraints: Dict[str, float], weights: Dict[str, float]) -> Optional[Dict[str, float]]:
        res = 25
        p_range = np.linspace(*self.domain["Pressure"], res)
        c_range = np.linspace(*self.domain["Conc"], res)
        ph_range = np.linspace(*self.domain["pH"], res)
        P, C, PH = np.meshgrid(p_range, c_range, ph_range)
        X_grid = np.column_stack([P.ravel(), C.ravel(), PH.ravel()])
        preds, _ = self.batch_predict_ensemble(X_grid)
        
        idx_flux = self.outputs.index("Flux")
        idx_rej = {name: self.outputs.index(name) for name in ["SMX", "TRM", "TET", "ERY"]}
        flux_vals = np.maximum(0.0, preds[:, idx_flux])
        min_r = constraints.get('min_rej', 0.0)
        min_f = constraints.get('min_flux', 0.0)
        
        valid_mask = (flux_vals >= min_f)
        for name in idx_rej:
            r_col = np.clip(preds[:, idx_rej[name]], 0.0, 100.0)
            valid_mask &= (r_col >= min_r)
        
        if not np.any(valid_mask): return None

        z_scores_raw = (preds - self.Y_mean.values) / self.Y_std.values
        scores = np.zeros(len(X_grid))
        w_flux = weights.get("Flux", 0.0)
        scores += w_flux * z_scores_raw[:, idx_flux]
        for r_name in ["SMX", "TRM", "TET", "ERY"]:
            w_r = weights.get(r_name, 0.0)
            scores += w_r * z_scores_raw[:, idx_rej[r_name]]

        final_scores = np.full(len(scores), -np.inf)
        final_scores[valid_mask] = scores[valid_mask]
        best_idx = np.argmax(final_scores)
        
        rej_cols_indices = [idx_rej[k] for k in idx_rej]
        avg_rej_vals = np.mean(preds[:, rej_cols_indices], axis=1)
        best_r = {name: np.clip(preds[best_idx, idx_rej[name]], 0, 100) for name in idx_rej}
        
        return {
            "Pressure": X_grid[best_idx, 0],
            "Conc": X_grid[best_idx, 1],
            "pH": X_grid[best_idx, 2],
            "Flux": flux_vals[best_idx],
            "Avg_Rej": avg_rej_vals[best_idx],
            "Rejections": best_r
        }

# ============================================================
# ------------------- 2. DATA MAPPINGS -----------------------
# ============================================================

LABELS_UI = {
    "Pressure": "Pressure (bar)",
    "Conc": "Concentration (mg/L)",
    "pH": "pH",
    "Flux": "Flux (LMH)",
    "SMX": "SMX Rejection (%)",
    "TRM": "TRM Rejection (%)",
    "TET": "TET Rejection (%)",
    "ERY": "ERY Rejection (%)"
}
UI_TO_KEY = {v: k for k, v in LABELS_UI.items()}
LABELS_PLOT = {
    "Pressure": "Pressure (bar)",
    "Conc": "Concentration (mg/L)",
    "pH": "pH",
    "Flux": "Flux (LMH)",
    "SMX": r"$R_{SMX}$ (%)",
    "TRM": r"$R_{TRM}$ (%)",
    "TET": r"$R_{TET}$ (%)",
    "ERY": r"$R_{ERY}$ (%)"
}

# ============================================================
# ------------------- 3. UI COMPONENTS -----------------------
# ============================================================

class MetricCard(ctk.CTkFrame):
    def __init__(self, parent, title, suffix="%", is_main=False):
        # Light Grey card background in Light mode, Dark Grey in Dark mode
        super().__init__(parent, fg_color=COLORS["bg_card"], corner_radius=10, border_width=1, border_color="#E5E7EB")
        self.suffix = suffix
        
        # Title Color: Black/White generally.
        title_color = COLORS["text_main"] 
        
        ctk.CTkLabel(self, text=title, font=("Cambria", 11, "bold"), 
                     text_color=title_color).pack(anchor="w", padx=8, pady=(4, 0))
        
        font_size = 22 if is_main else 18
        
        # Value Color: Green if Flux, else Violet/Light Violet
        if "Flux" in title:
            color = COLORS["primary"]
        else:
            color = COLORS["accent"] if is_main else COLORS["text_main"]
            
        self.value_label = ctk.CTkLabel(self, text=f"0.0{suffix}", 
                                        font=("Cambria", font_size, "bold"), text_color=color)
        self.value_label.pack(anchor="w", padx=8, pady=(0, 4))
        
        if not is_main:
            self.progress = ctk.CTkProgressBar(self, height=4)
            self.progress.pack(fill="x", padx=8, pady=(0, 8))
            self.progress.set(0)

    def update_value(self, value, std=None):
        if std is not None and std > 0:
            self.value_label.configure(text=f"{value:.1f} ± {std:.1f}{self.suffix}")
        else:
            self.value_label.configure(text=f"{value:.1f}{self.suffix}")
            
        if hasattr(self, 'progress'):
            self.progress.set(value / 100)
            if value >= 90: color = "#10B981" # Green
            elif value >= 70: color = "#F59E0B" # Orange
            else: color = "#EF4444" # Red
            self.progress.configure(progress_color=color)

class WorkerJob:
    def __init__(self, target, args=(), kwargs=None, progress_cb=None, done_cb=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.progress_cb = progress_cb
        self.done_cb = done_cb
        self._thread = None
        self._cancel = False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel = True

    def _run(self):
        try:
            result = self.target(*self.args, cancel_flag=lambda: self._cancel, progress_cb=self.progress_cb, **self.kwargs)
            if self.done_cb:
                self.done_cb(result)
        except Exception as e:
            if self.done_cb:
                self.done_cb(e)


# ============================================================
# ------------------- 4. UI CONFIGURATION --------------------
# ============================================================

class MembraneApp(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=COLORS["bg_main"])
        
        try:
            self.engine = ScientificEngine()
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            self.destroy()
            return

        self.title("H-SPES Membrane Simulator")
        self.geometry("1200x750")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.debounce_job = None
        self.ignore_weight_change = False
        self.current_surface_job = None
        
        self._prog_win = None
        self._prog_bar = None

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=COLORS["bg_card"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.setup_sidebar()

        # Main Area
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.setup_dashboard()
        
        # Tabs
        self.plot_tabs = ctk.CTkTabview(self.main_area, fg_color=COLORS["bg_card"], text_color=COLORS["text_main"])
        self.plot_tabs.pack(fill="both", expand=True, pady=(8, 0))
        self.plot_tabs._segmented_button.configure(font=("Cambria", 12, "bold"))
        self.plot_tabs.add("2D Sensitivity")
        self.plot_tabs.add("3D Surface Analysis")
        self.plot_tabs.add("Batch Prediction")
        self.plot_tabs.add("Optimizer")
        
        self.setup_plot_2d()
        self.setup_plot_3d()
        self.setup_batch_tab()
        self.setup_opt_tab()

        self.run_simulation()

    def setup_sidebar(self):
        # Header (Fixed - No Toggle)
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(fill="x", pady=(12, 4))
        
        ctk.CTkLabel(header_frame, text="H-SPES SIMULATOR", font=("Cambria", 16, "bold"), text_color=COLORS["text_main"]).pack(padx=12)

        # Scenarios Frame
        sc_frame = ctk.CTkFrame(self.sidebar, fg_color=COLORS["bg_main"])
        sc_frame.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(sc_frame, text="Save Scenario", width=80, height=22, font=("Cambria", 10), 
                      fg_color=COLORS["secondary"], text_color=COLORS["text_main"], hover_color=COLORS["secondary_hov"],
                      command=self.save_scenario).pack(side="left", padx=4, pady=4)
        ctk.CTkButton(sc_frame, text="Load Scenario", width=80, height=22, font=("Cambria", 10), 
                      fg_color=COLORS["secondary"], text_color=COLORS["text_main"], hover_color=COLORS["secondary_hov"],
                      command=self.load_scenario).pack(side="right", padx=4, pady=4)

        self.add_section_header("MODEL CONFIGURATION")
        self.model_var = ctk.StringVar(value="Ensemble")
        ctk.CTkOptionMenu(self.sidebar, variable=self.model_var, 
                          values=["Ensemble", "ReLU", "Tanh", "Logistic"],
                          fg_color=COLORS["primary"], button_color=COLORS["primary_dark"], text_color=COLORS["text_main"],
                          command=self.run_simulation, font=("Cambria", 11), height=22).pack(fill="x", padx=12, pady=2)

        self.add_section_header("OPERATING CONDITIONS")
        self.inputs = {}
        self.create_slider(LABELS_UI["Pressure"], "p", 0.5, 3.5, 2.0)
        self.create_slider(LABELS_UI["Conc"], "c", 2.0, 10.0, 6.0)
        self.create_slider(LABELS_UI["pH"], "ph", 5.0, 9.0, 7.0)

        self.add_section_header("PLOT AXIS CONTROLS")
        ctk.CTkLabel(self.sidebar, text="Target Output:", font=("Cambria", 11), text_color=COLORS["text_main"]).pack(anchor="w", padx=12)
        self.target_var = ctk.StringVar(value=LABELS_UI["Flux"])
        vals = [LABELS_UI[k] for k in ["Flux", "SMX", "TRM", "TET", "ERY"]]
        ctk.CTkOptionMenu(self.sidebar, variable=self.target_var, values=vals,
                          fg_color=COLORS["primary"], button_color=COLORS["primary_dark"], text_color=COLORS["text_main"],
                          command=self.run_simulation, font=("Cambria", 11), height=22).pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(self.sidebar, text="X-Axis Input:", font=("Cambria", 11), text_color=COLORS["text_main"]).pack(anchor="w", padx=12)
        self.xaxis_var = ctk.StringVar(value=LABELS_UI["Pressure"])
        in_vals = [LABELS_UI[k] for k in ["Pressure", "Conc", "pH"]]
        ctk.CTkOptionMenu(self.sidebar, variable=self.xaxis_var, values=in_vals,
                          fg_color=COLORS["primary"], button_color=COLORS["primary_dark"], text_color=COLORS["text_main"],
                          command=self.run_simulation, font=("Cambria", 11), height=22).pack(fill="x", padx=12, pady=(0, 4))
        
        ctk.CTkLabel(self.sidebar, text="Y-Axis Input (3D):", font=("Cambria", 11), text_color=COLORS["text_main"]).pack(anchor="w", padx=12)
        self.y3d_var = ctk.StringVar(value=LABELS_UI["Conc"])
        ctk.CTkOptionMenu(self.sidebar, variable=self.y3d_var, values=in_vals,
                          fg_color=COLORS["primary"], button_color=COLORS["primary_dark"], text_color=COLORS["text_main"],
                          command=self.run_simulation, font=("Cambria", 11), height=22).pack(fill="x", padx=12, pady=(0, 4))

        self.add_section_header("VISUALIZATION")
        self.res_slider = ctk.CTkSlider(self.sidebar, from_=10, to=100, number_of_steps=90, command=self.on_res_change, height=12,
                                        button_color=COLORS["primary"], progress_color=COLORS["primary"])
        self.res_slider.set(25)
        self.res_slider.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(self.sidebar, text="Grid Resolution", font=("Cambria", 10), text_color=COLORS["text_sub"]).pack(anchor="center")

        btn_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=12, side="bottom")
        ctk.CTkButton(btn_row, text="SAVE FIGURES", height=26, width=90, command=self.save_plots,
                      font=("Cambria", 10, "bold"), fg_color=COLORS["secondary"], hover_color=COLORS["secondary_hov"], text_color=COLORS["text_main"]).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row, text="RESET", height=26, width=90, command=self.reset_defaults,
                      font=("Cambria", 10, "bold"), fg_color=COLORS["destructive"], hover_color=COLORS["destructive_h"], text_color=COLORS["text_main"]).pack(side="right", padx=(4, 0))

    def add_section_header(self, text):
        ctk.CTkLabel(self.sidebar, text=text, font=("Cambria", 11, "bold"), text_color=COLORS["text_sub"]).pack(anchor="w", padx=12, pady=(8, 2))

    def create_slider(self, label, key, min_val, max_val, default):
        frame = ctk.CTkFrame(self.sidebar, fg_color=COLORS["bg_main"])
        frame.pack(fill="x", padx=12, pady=2)
        
        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x", padx=4, pady=(2,0))
        ctk.CTkLabel(top, text=label, anchor="w", font=("Cambria", 11), text_color=COLORS["text_main"]).pack(side="left")
        
        val_entry = ctk.CTkEntry(top, width=50, height=20, font=("Cambria", 11), text_color=COLORS["text_main"], fg_color=COLORS["input_bg"])
        val_entry.pack(side="right")
        val_entry.insert(0, f"{default:.1f}")
        
        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, number_of_steps=100, height=12,
                               command=lambda v: self.on_slider_drag(v, key, val_entry),
                               button_color=COLORS["primary"], progress_color=COLORS["primary"])
        slider.set(default)
        slider.pack(fill="x", padx=4, pady=(0, 4))
        
        val_entry.bind("<Return>", lambda e: self.on_entry_commit(key, val_entry, slider, min_val, max_val))
        val_entry.bind("<FocusOut>", lambda e: self.on_entry_commit(key, val_entry, slider, min_val, max_val))
        self.inputs[key] = {"slider": slider, "entry": val_entry, "val": default, "frame": frame}

    def setup_dashboard(self):
        self.flux_card = MetricCard(self.main_area, "Permeate Flux", " LMH", is_main=True)
        self.flux_card.pack(fill="x", pady=(0, 8))
        
        self.warn_lbl = ctk.CTkLabel(self.flux_card, text="⚠️ Extrapolating", font=("Cambria", 11, "bold"), text_color="#F59E0B")

        grid = ctk.CTkFrame(self.main_area, fg_color="transparent")
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1)
        
        self.rej_widgets = {}
        metrics = [("SMX", 0, 0), ("TRM", 0, 1), ("TET", 1, 0), ("ERY", 1, 1)]
        
        for key, r, c in metrics:
            card = MetricCard(grid, LABELS_UI[key])
            card.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            self.rej_widgets[key] = card

    def setup_plot_2d(self):
        self.fig_2d, self.ax_2d = plt.subplots(figsize=(5, 2.5), dpi=100)
        self.fig_2d.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.22)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.plot_tabs.tab("2D Sensitivity"))
        self.canvas_2d.get_tk_widget().pack(fill="both", expand=True)

    def setup_plot_3d(self):
        self.fig_3d = plt.figure(figsize=(9, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(121, projection='3d')
        self.ax_contour = self.fig_3d.add_subplot(122)
        self.fig_3d.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.94, wspace=0.5)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.plot_tabs.tab("3D Surface Analysis"))
        self.canvas_3d.get_tk_widget().pack(fill="both", expand=True)

    def setup_opt_tab(self):
        tab = self.plot_tabs.tab("Optimizer")
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        
        ctk.CTkLabel(tab, text="Operating Window Optimizer", font=("Cambria", 16, "bold"), text_color=COLORS["text_main"]).grid(row=0, column=0, columnspan=2, pady=(8, 4))

        left_frame = ctk.CTkFrame(tab, fg_color="transparent")
        left_frame.grid(row=1, column=0, padx=8, pady=4, sticky="n")
        
        ctk.CTkLabel(left_frame, text="Optimization Goal:", font=("Cambria", 13, "bold"), text_color=COLORS["text_main"]).pack(anchor="w", pady=(0, 2))
        self.opt_obj_var = ctk.StringVar(value="Maximize Flux")
        
        self.opt_mode_menu = ctk.CTkOptionMenu(left_frame, variable=self.opt_obj_var, 
                                               values=["Maximize Flux", "Maximize SMX Rejection", "Maximize TRM Rejection", 
                                                       "Maximize TET Rejection", "Maximize ERY Rejection", "Custom Weighted"],
                                               fg_color=COLORS["primary"], button_color=COLORS["primary_dark"], text_color=COLORS["text_main"],
                                               command=self.on_opt_mode_change, font=("Cambria", 11), width=200, height=26)
        self.opt_mode_menu.pack(pady=4)
        
        self.weights_frame = ctk.CTkFrame(left_frame, fg_color=COLORS["bg_main"])
        self.weights_frame.pack(fill="x", pady=8, ipady=2)
        ctk.CTkLabel(self.weights_frame, text="Weights (Impact Factor)", font=("Cambria", 11, "bold"), text_color=COLORS["text_main"]).pack(pady=2)
        
        w_grid = ctk.CTkFrame(self.weights_frame, fg_color="transparent")
        w_grid.pack(pady=2)
        
        self.weight_entries = {}
        vars_list = ["Flux", "SMX", "TRM", "TET", "ERY"]
        for i, v in enumerate(vars_list):
            row = i // 2
            col = (i % 2) * 2
            ctk.CTkLabel(w_grid, text=f"{v}:", font=("Cambria", 10), text_color=COLORS["text_main"]).grid(row=row, column=col, padx=4, pady=1, sticky="e")
            e = ctk.CTkEntry(w_grid, width=40, height=22, fg_color=COLORS["input_bg"], text_color=COLORS["text_main"])
            if v == "Flux": e.insert(0, "1.0")
            else: e.insert(0, "0.0")
            e.grid(row=row, column=col+1, padx=4, pady=1)
            e.bind("<KeyRelease>", self.on_manual_weight_edit)
            self.weight_entries[v] = e

        con_frame = ctk.CTkFrame(left_frame, fg_color=COLORS["bg_main"])
        con_frame.pack(fill="x", pady=8, ipady=2)
        ctk.CTkLabel(con_frame, text="Constraints", font=("Cambria", 11, "bold"), text_color=COLORS["text_main"]).pack(pady=2)
        
        grid_con = ctk.CTkFrame(con_frame, fg_color="transparent")
        grid_con.pack()
        
        ctk.CTkLabel(grid_con, text="Min Flux (LMH):", font=("Cambria", 10), text_color=COLORS["text_main"]).grid(row=0, column=0, padx=6, pady=2, sticky="e")
        self.opt_min_flux = ctk.CTkEntry(grid_con, width=45, height=22, fg_color=COLORS["input_bg"], text_color=COLORS["text_main"])
        self.opt_min_flux.insert(0, "20")
        self.opt_min_flux.grid(row=0, column=1, padx=6, pady=2)
        
        ctk.CTkLabel(grid_con, text="Min Rejection (%):", font=("Cambria", 10), text_color=COLORS["text_main"]).grid(row=1, column=0, padx=6, pady=2, sticky="e")
        self.opt_min_rej = ctk.CTkEntry(grid_con, width=45, height=22, fg_color=COLORS["input_bg"], text_color=COLORS["text_main"])
        self.opt_min_rej.insert(0, "25")
        self.opt_min_rej.grid(row=1, column=1, padx=6, pady=2)

        ctk.CTkButton(left_frame, text="FIND OPTIMAL CONDITIONS", command=self.run_optimizer_async, height=36, width=200,
                      font=("Cambria", 12, "bold"), fg_color=COLORS["primary"], hover_color=COLORS["primary_dark"], text_color=COLORS["text_main"]).pack(pady=12)
        
        right_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"])
        right_frame.grid(row=1, column=1, padx=8, pady=4, sticky="nsew")
        
        ctk.CTkLabel(right_frame, text="Optimization Results", font=("Cambria", 14, "bold"), text_color=COLORS["text_main"]).pack(pady=8)
        
        self.opt_result_txt = ctk.CTkTextbox(right_frame, width=260, height=280, font=("Consolas", 11), 
                                             fg_color=COLORS["bg_main"], text_color=COLORS["text_main"], border_width=1, border_color="#E5E7EB")
        self.opt_result_txt.pack(pady=4, padx=12)
        self.opt_result_txt.insert("0.0", "Configure settings and click 'Find Optimal Conditions'.")
        self.opt_result_txt.configure(state="disabled")

    # --- LOGIC ---

    def on_opt_mode_change(self, mode):
        self.ignore_weight_change = True
        new_weights = {"Flux": 0.0, "SMX": 0.0, "TRM": 0.0, "TET": 0.0, "ERY": 0.0}
        if mode == "Maximize Flux": new_weights["Flux"] = 1.0
        elif mode == "Maximize SMX Rejection": new_weights["SMX"] = 1.0
        elif mode == "Maximize TRM Rejection": new_weights["TRM"] = 1.0
        elif mode == "Maximize TET Rejection": new_weights["TET"] = 1.0
        elif mode == "Maximize ERY Rejection": new_weights["ERY"] = 1.0
        
        if mode != "Custom Weighted":
            for k, val in new_weights.items():
                entry = self.weight_entries[k]
                entry.delete(0, "end")
                entry.insert(0, f"{val:.1f}")
        self.ignore_weight_change = False

    def on_manual_weight_edit(self, event):
        if not self.ignore_weight_change:
            self.opt_obj_var.set("Custom Weighted")

    def run_optimizer_async(self):
        self._show_progress("Optimizing...")
        try:
            min_r = float(self.opt_min_rej.get())
            min_f = float(self.opt_min_flux.get())
            w_dict = {}
            for k, ent in self.weight_entries.items():
                w_dict[k] = float(ent.get())
            total = sum(w_dict.values())
            if total == 0: raise ValueError("Total weight cannot be zero.")
            if abs(total - 1.0) > 1e-6:
                for k in w_dict: w_dict[k] /= total
            
            def optimize_task(cancel_flag, progress_cb):
                return self.engine.find_optimum({'min_rej': min_r, 'min_flux': min_f}, w_dict)

            def on_done(res):
                self.after(0, lambda: self._handle_opt_done(res, w_dict))
            job = WorkerJob(target=optimize_task, done_cb=on_done)
            job.start()
        except Exception as e:
            self._hide_progress()
            messagebox.showerror("Optimizer Error", str(e))

    def _handle_opt_done(self, res, w_dict):
        self._hide_progress()
        if isinstance(res, Exception):
            messagebox.showerror("Optimizer Error", str(res))
        else:
            self._display_optimizer_result(res, w_dict)

    def _display_optimizer_result(self, res, w_dict):
        self.opt_result_txt.configure(state="normal")
        self.opt_result_txt.delete("0.0", "end")
        self.ignore_weight_change = True
        for k in w_dict:
            self.weight_entries[k].delete(0, "end")
            self.weight_entries[k].insert(0, f"{w_dict[k]:.3f}")
        self.ignore_weight_change = False

        if res:
            rej_str = "\n".join([f"    {k}: {v:.1f}%" for k, v in res['Rejections'].items()])
            txt = (f"✅ OPTIMAL CONDITIONS FOUND\n\n"
                   f"• Pressure:  {res['Pressure']:.2f} bar\n"
                   f"• Conc:      {res['Conc']:.2f} mg/L\n"
                   f"• pH:        {res['pH']:.2f}\n\n"
                   f"PREDICTED PERFORMANCE:\n"
                   f"• Flux:      {res['Flux']:.2f} LMH\n"
                   f"• Avg Rej:   {res['Avg_Rej']:.2f}%\n"
                   f"• Detailed Rejections:\n{rej_str}")
            self.opt_result_txt.insert("0.0", txt)
        else:
            self.opt_result_txt.insert("0.0", "❌ No conditions found satisfying these constraints.\nTry lowering the minimum rejection or flux requirements.")
        self.opt_result_txt.configure(state="disabled")

    # --- Batch Tab ---
    def setup_batch_tab(self):
        tab = self.plot_tabs.tab("Batch Prediction")
        ctk.CTkLabel(tab, text="Batch Prediction Mode", font=("Cambria", 16, "bold"), text_color=COLORS["text_main"]).pack(pady=(15, 4))
        ctk.CTkLabel(tab, text="Upload CSV with columns: Pressure (bar), Conc (mg/L), pH", 
                     font=("Cambria", 11), text_color=COLORS["text_sub"]).pack(pady=(0, 15))
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.pack()
        ctk.CTkButton(btn_frame, text="Generate Template", command=self.generate_template,
                      font=("Cambria", 11, "bold"), height=32, width=160, fg_color=COLORS["primary"], text_color="black").pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Load CSV & Run", command=self.start_batch_thread,
                      font=("Cambria", 11, "bold"), height=32, width=160, fg_color=COLORS["secondary"], text_color="black").pack(side="left", padx=8)

    def generate_template(self):
        f = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="batch_template.csv")
        if f:
            data = [
                [0.5, 2.0, 7.0], [3.5, 2.0, 7.0], [0.5, 10.0, 7.0], [3.5, 10.0, 7.0],
                [0.5, 6.0, 5.0], [3.5, 6.0, 5.0], [0.5, 6.0, 9.0], [3.5, 6.0, 9.0],
                [2.0, 2.0, 5.0], [2.0, 10.0, 5.0], [2.0, 2.0, 9.0], [2.0, 10.0, 9.0],
                [2.0, 6.0, 7.0]
            ]
            pd.DataFrame(data, columns=["Pressure (bar)", "Conc (mg/L)", "pH"]).to_csv(f, index=False)
            messagebox.showinfo("Done", "Template saved.\nDomain: P 0.5-3.5, C 2-10, pH 5-9")

    def start_batch_thread(self):
        f = filedialog.askopenfilename()
        if f: threading.Thread(target=self.run_batch_prediction, args=(f,), daemon=True).start()

    def run_batch_prediction(self, f):
        try:
            df = pd.read_csv(f)
            cols = [c.split(" (")[0] for c in df.columns]
            df.columns = cols
            req = ["Pressure", "Conc", "pH"]
            missing = [c for c in req if c not in df.columns]
            if missing:
                raise ValueError(f"Input file is missing required columns: {', '.join(missing)}\nExpected columns (case sensitive): Pressure, Conc, pH")
            try:
                df[req] = df[req].apply(pd.to_numeric)
            except Exception:
                raise ValueError("All input columns must be numeric.")

            pred, std = self.engine.batch_predict_ensemble(df[req].values)
            extrap_flags = []
            for row in df[req].itertuples(index=False, name=None):
                extrap_flags.append(self.engine.is_extrapolating(*row))
            df["Extrapolating"] = extrap_flags

            for i, n in enumerate(self.engine.outputs):
                df[f"Pred_{n}"] = np.round(pred[:, i], 2)
                df[f"Uncertainty_{n}"] = np.round(std[:, i], 2)
            self.after(0, lambda: self.save_batch_result(df))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    def save_batch_result(self, df):
        f = filedialog.asksaveasfilename(defaultextension=".csv")
        if f:
            df.to_csv(f, index=False)
            messagebox.showinfo("Success", "Saved.")

    # --- Common Logic ---
    def on_slider_drag(self, v, k, e):
        self.inputs[k]["val"] = v
        e.delete(0, "end")
        e.insert(0, f"{v:.1f}")
        if self.debounce_job: self.after_cancel(self.debounce_job)
        self.debounce_job = self.after(100, self.run_simulation)

    def on_entry_commit(self, k, e, s, min_v, max_v):
        try:
            v = float(e.get())
            v = max(min_v, min(v, max_v))
            s.set(v)
            self.inputs[k]["val"] = v
            self.run_simulation()
        except: pass

    def on_res_change(self, v):
        if self.debounce_job: self.after_cancel(self.debounce_job)
        self.debounce_job = self.after(200, self.run_simulation)

    def highlight_extrapolation(self, extrapolating: bool):
        color = COLORS["warn_light"] if extrapolating else COLORS["bg_main"]
        for k in ["p", "c", "ph"]:
            frame = self.inputs[k]["frame"]
            frame.configure(fg_color=color)

    def reset_defaults(self):
        defaults = {"p": 2.0, "c": 6.0, "ph": 7.0}
        for key, val in defaults.items():
            self.inputs[key]["val"] = val
            self.inputs[key]["slider"].set(val)
            self.inputs[key]["entry"].delete(0, "end")
            self.inputs[key]["entry"].insert(0, f"{val:.1f}")
        
        self.model_var.set("Ensemble")
        self.target_var.set(LABELS_UI["Flux"])
        self.xaxis_var.set(LABELS_UI["Pressure"])
        self.y3d_var.set(LABELS_UI["Conc"])
        self.res_slider.set(25)
        
        self.opt_min_flux.delete(0, "end"); self.opt_min_flux.insert(0, "20")
        self.opt_min_rej.delete(0, "end"); self.opt_min_rej.insert(0, "25")
        self.opt_obj_var.set("Maximize Flux")
        self.on_opt_mode_change("Maximize Flux")
        self.opt_result_txt.configure(state="normal")
        self.opt_result_txt.delete("0.0", "end")
        self.opt_result_txt.insert("0.0", "Configure settings and click 'Find Optimal Conditions'.")
        self.opt_result_txt.configure(state="disabled")

        self.run_simulation()
        self.ax_3d.view_init(elev=30, azim=-60)
        self.canvas_3d.draw()

    def save_plots(self):
        try:
            path_2d = filedialog.asksaveasfilename(defaultextension=".png", initialfile="Sensitivity_2D.png", title="Save 2D Plot")
            if path_2d: self.fig_2d.savefig(path_2d, dpi=300, bbox_inches='tight', facecolor="white")
            path_3d = filedialog.asksaveasfilename(defaultextension=".png", initialfile="Surface_3D.png", title="Save 3D Plot")
            if path_3d: self.fig_3d.savefig(path_3d, dpi=300, bbox_inches='tight', facecolor="white")
            messagebox.showinfo("Success", "Plots saved!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save plots: {e}")

    def save_scenario(self):
        d = {
            "p": self.inputs["p"]["val"],
            "c": self.inputs["c"]["val"],
            "ph": self.inputs["ph"]["val"],
            "m": self.model_var.get(),
            "t": self.target_var.get(),
            "x": self.xaxis_var.get(),
            "y": self.y3d_var.get()
        }
        f = filedialog.asksaveasfilename(defaultextension=".json")
        if f:
            with open(f, 'w') as o: json.dump(d, o)
            messagebox.showinfo("Saved", "Scenario saved.")

    def load_scenario(self):
        f = filedialog.askopenfilename()
        if f:
            try:
                with open(f, 'r') as i: d = json.load(i)
                for k in ["p","c","ph"]:
                    self.inputs[k]["val"] = d[k]
                    self.inputs[k]["slider"].set(d[k])
                    self.inputs[k]["entry"].delete(0,"end")
                    self.inputs[k]["entry"].insert(0, f"{d[k]:.1f}")
                self.model_var.set(d["m"])
                self.target_var.set(d["t"])
                self.xaxis_var.set(d["x"])
                self.y3d_var.set(d["y"])
                self.run_simulation()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load scenario: {e}")

    # --- Progress Modal Logic ---
    def _show_progress(self, title="Working..."):
        if self._prog_win is not None and self._prog_win.winfo_exists():
            return
        self._prog_win = ctk.CTkToplevel(self)
        self._prog_win.title(title)
        self._prog_win.geometry("280x90")
        self._prog_win.transient(self)
        self._prog_win.grab_set() 
        ctk.CTkLabel(self._prog_win, text=title, font=("Cambria", 11)).pack(pady=(8, 4))
        self._prog_bar = ctk.CTkProgressBar(self._prog_win, mode="determinate", height=6, progress_color=COLORS["primary"])
        self._prog_bar.set(0)
        self._prog_bar.pack(fill="x", padx=15, pady=4)
        ctk.CTkButton(self._prog_win, text="Cancel", command=self._cancel_current_job, 
                      fg_color=COLORS["destructive"], hover_color=COLORS["destructive_h"], text_color="black", height=22).pack(pady=(4,8))

    def _update_progress(self, pct):
        self.after(0, lambda: self._do_update_progress(pct))

    def _do_update_progress(self, pct):
        if self._prog_bar is not None and self._prog_bar.winfo_exists():
            self._prog_bar.set(pct / 100.0)

    def _hide_progress(self):
        if self._prog_win is not None and self._prog_win.winfo_exists():
            try:
                self._prog_win.grab_release()
                self._prog_win.withdraw()
                self._prog_win.destroy()
            except Exception: pass
        self._prog_win = None
        self._prog_bar = None

    def _cancel_current_job(self):
        if getattr(self, "current_surface_job", None) is not None:
            self.current_surface_job.cancel()

    @lru_cache(maxsize=32)
    def _get_cached_surface_grid(self, x_key, y_key, res, p_val, c_val, ph_val):
        ranges = {
            "Pressure": np.linspace(0.5, 3.5, res),
            "Conc":     np.linspace(2.0, 10.0, res),
            "pH":       np.linspace(5.0, 9.0, res)
        }
        X_grid, Y_grid = np.meshgrid(ranges[x_key], ranges[y_key])
        N = res * res
        X_batch = np.zeros((N, 3))
        col_map = {"Pressure": 0, "Conc": 1, "pH": 2}
        X_batch[:, 0] = p_val
        X_batch[:, 1] = c_val
        X_batch[:, 2] = ph_val
        X_batch[:, col_map[x_key]] = X_grid.ravel()
        X_batch[:, col_map[y_key]] = Y_grid.ravel()
        return X_grid, Y_grid, X_batch

    def build_X_batch_2d(self, base_p, base_c, base_ph, varied_name, varied_values):
        n = len(varied_values)
        X = np.zeros((n, 3))
        X[:, 0] = base_p
        X[:, 1] = base_c
        X[:, 2] = base_ph
        col_map = {"Pressure": 0, "Conc": 1, "pH": 2}
        X[:, col_map[varied_name]] = varied_values
        return X

    def run_simulation(self, _=None):
        p, c, ph = self.inputs["p"]["val"], self.inputs["c"]["val"], self.inputs["ph"]["val"]
        model = self.model_var.get()
        res = self.engine.predict(p, c, ph, model)
        
        extrap = res.get("extrapolating")
        if extrap: self.warn_lbl.place(relx=0.5, rely=0.85, anchor="center")
        else: self.warn_lbl.place_forget()
        self.highlight_extrapolation(extrap)

        fl = res['Flux']
        fl_std = res.get('Flux_std', 0)
        self.flux_card.update_value(fl, fl_std if model == "Ensemble" else None)
        
        for k, w in self.rej_widgets.items():
            val = res[k]
            std = res.get(f"{k}_std", 0)
            w.update_value(val, std if model == "Ensemble" else None)

        self.update_plot_2d(p, c, ph, model, res)
        self.start_surface_job(p, c, ph, model)

    def update_plot_2d(self, current_p, current_c, current_ph, model, current_res):
        x_target = UI_TO_KEY[self.xaxis_var.get()]
        y_target = UI_TO_KEY[self.target_var.get()]
        
        ranges = {
            "Pressure": (np.linspace(0.5, 3.5, 100), "bar"),
            "Conc":     (np.linspace(2.0, 10.0, 100), "ppm"),
            "pH":       (np.linspace(5.0, 9.0, 100), "")
        }
        x_vals, unit = ranges[x_target]
        X_batch = self.build_X_batch_2d(current_p, current_c, current_ph, x_target, x_vals)
        
        if model == "Ensemble":
            pred_raw, std_raw = self.engine.batch_predict_ensemble(X_batch)
            out_idx = self.engine.outputs.index(y_target)
            y_vals = pred_raw[:, out_idx]
            y_stds = std_raw[:, out_idx]
            if y_target == "Flux": y_vals = np.maximum(0.0, y_vals)
            else: y_vals = np.clip(y_vals, 0.0, 100.0)
        else:
            x_z = ((X_batch - self.engine.X_mean.values) / self.engine.X_std.values)
            mdl = self.engine.models.get(model.lower())
            pz = mdl.predict(x_z)
            if pz.ndim == 1: pz = pz.reshape(-1, 1)
            p_raw = pz * self.engine.Y_std.values + self.engine.Y_mean.values
            out_idx = self.engine.outputs.index(y_target)
            y_vals = p_raw[:, out_idx]
            y_stds = np.zeros_like(y_vals)
            if y_target == "Flux": y_vals = np.maximum(0.0, y_vals)
            else: y_vals = np.clip(y_vals, 0.0, 100.0)

        # Matplotlib Colors (Always Light)
        bg_col = "#F9FAFB"
        fg_col = "black"
        grid_col = "#E5E5E5"
        line_col = COLORS["secondary"][0]

        self.ax_2d.clear()
        self.ax_2d.set_facecolor(bg_col)
        self.ax_2d.set_ylim(0, 100)
        
        if model == "Ensemble":
            fill_col = COLORS["primary"][0]
            self.ax_2d.fill_between(x_vals, y_vals - y_stds, y_vals + y_stds, color=fill_col, alpha=0.2, linewidth=0)
            
        self.ax_2d.plot(x_vals, y_vals, color=line_col, linewidth=2.5) 
        
        current_y = current_res[y_target]
        current_x = current_p if x_target == "Pressure" else current_c if x_target == "Conc" else current_ph
        
        dot_edge = COLORS["accent"][0]
        self.ax_2d.scatter([current_x], [current_y], color='white', s=60, zorder=5, edgecolors=dot_edge)

        if self.engine.experimental_data is not None:
            df = self.engine.experimental_data
            mask = np.ones(len(df), dtype=bool)
            current_vals = {"Pressure": current_p, "Conc": current_c, "pH": current_ph}
            for inp in self.engine.inputs:
                if inp != x_target:
                    mask &= (np.abs(df[inp] - current_vals[inp]) < 0.1)
            df_filtered = df[mask]
            if not df_filtered.empty:
                ex_x = df_filtered[x_target]
                ex_y = df_filtered[y_target]
                self.ax_2d.scatter(ex_x, ex_y, color='white', edgecolors='black', s=60, label="Experimental Data", zorder=10)

        self.ax_2d.set_xlabel(LABELS_PLOT[x_target], color=fg_col, fontsize=14) 
        self.ax_2d.set_ylabel(LABELS_PLOT[y_target], color=fg_col, fontsize=14) 
        self.ax_2d.tick_params(colors=fg_col, labelsize=14) 
        self.ax_2d.spines['bottom'].set_color(fg_col) 
        self.ax_2d.spines['left'].set_color(fg_col) 
        self.ax_2d.spines['top'].set_visible(False)
        self.ax_2d.spines['right'].set_visible(False)
        self.ax_2d.grid(True, color=grid_col, linestyle='--')
        self.fig_2d.tight_layout()
        self.canvas_2d.draw()

    def start_surface_job(self, current_p, current_c, current_ph, model):
        x_key = UI_TO_KEY[self.xaxis_var.get()]
        y_key = UI_TO_KEY[self.y3d_var.get()]
        z_key = UI_TO_KEY[self.target_var.get()]
        res = int(self.res_slider.get())
        if self.current_surface_job: self.current_surface_job.cancel()
        show_prog = (res > 40)
        if show_prog: self._show_progress("Computing 3D Surface...")

        def compute_surface(cancel_flag, progress_cb):
            X_grid, Y_grid, X_batch = self._get_cached_surface_grid(x_key, y_key, res, current_p, current_c, current_ph)
            N = X_batch.shape[0]
            chunk_size = 2000
            preds = []
            stds = []
            for i in range(0, N, chunk_size):
                if cancel_flag(): return None 
                end = min(i + chunk_size, N)
                if model == "Ensemble":
                    p_c, s_c = self.engine.batch_predict_ensemble(X_batch[i:end])
                else:
                    x_z = ((X_batch[i:end] - self.engine.X_mean.values) / self.engine.X_std.values)
                    mdl = self.engine.models.get(model.lower())
                    pz = mdl.predict(x_z)
                    if pz.ndim == 1: pz = pz.reshape(-1, 1)
                    p_c = pz * self.engine.Y_std.values + self.engine.Y_mean.values
                    s_c = np.zeros_like(p_c)
                preds.append(p_c)
                stds.append(s_c)
                if progress_cb: progress_cb(min(100, int(100 * end / N)))
            return np.vstack(preds), np.vstack(stds), X_grid, Y_grid

        def on_surface_done(result):
            self.after(0, lambda: self._handle_surface_done(result, x_key, y_key, z_key, model, res, current_p, current_c, current_ph, show_prog))
        self.current_surface_job = WorkerJob(target=compute_surface, progress_cb=self._update_progress if show_prog else None, done_cb=on_surface_done)
        self.current_surface_job.start()

    def _handle_surface_done(self, result, x_key, y_key, z_key, model, res, current_p, current_c, current_ph, show_prog):
        if show_prog: self._hide_progress()
        if result is None: return 
        if isinstance(result, Exception):
            messagebox.showerror("Error", str(result))
            return

        preds, stds, X_grid, Y_grid = result
        out_idx = self.engine.outputs.index(z_key)
        Z_flat = preds[:, out_idx]
        if z_key == "Flux": Z_flat = np.maximum(0.0, Z_flat)
        else: Z_flat = np.clip(Z_flat, 0.0, 100.0)
        Z_grid = Z_flat.reshape(res, res)
        Z_std_grid = None
        if model == "Ensemble":
            Z_std_grid = stds[:, out_idx].reshape(res, res)
        self._draw_3d_plot(X_grid, Y_grid, Z_grid, Z_std_grid, x_key, y_key, z_key, model, current_p, current_c, current_ph)

    def _draw_3d_plot(self, X_grid, Y_grid, Z_grid, Z_std_grid, x_key, y_key, z_key, model, current_p, current_c, current_ph):
        bg_col = "#F9FAFB"
        fg_col = "black"
        
        self.ax_3d.clear()
        self.ax_3d.set_facecolor(bg_col) 
        self.ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax_3d.grid(False)
        self.ax_3d.set_box_aspect((1, 1, 0.8))

        self.ax_3d.plot_surface(Y_grid, X_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9)
        
        mesh_col = "black"
        
        if Z_std_grid is not None:
             self.ax_3d.plot_surface(Y_grid, X_grid, Z_grid + Z_std_grid, color=mesh_col, alpha=0.1, linewidth=0, antialiased=False)
             self.ax_3d.plot_surface(Y_grid, X_grid, Z_grid - Z_std_grid, color=mesh_col, alpha=0.1, linewidth=0, antialiased=False)

        if x_key == y_key:
             txt_col = "red"
             self.ax_3d.text2D(0.05, 0.95, "2D Projection Mode", transform=self.ax_3d.transAxes, color=txt_col, fontsize=14) 

        if self.engine.experimental_data is not None:
            df = self.engine.experimental_data
            mask = np.ones(len(df), dtype=bool)
            current_vals = {"Pressure": current_p, "Conc": current_c, "pH": current_ph}
            active_axes = {x_key, y_key}
            for inp in self.engine.inputs:
                if inp not in active_axes:
                    mask &= (np.abs(df[inp] - current_vals[inp]) < 0.1)
            df_filtered = df[mask]
            if not df_filtered.empty:
                ex_x = df_filtered[x_key]
                ex_y = df_filtered[y_key]
                ex_z = df_filtered[z_key]
                self.ax_3d.scatter(ex_y, ex_x, ex_z, color='black', s=100, edgecolors='white', marker='o', alpha=1.0, zorder=10)

        self.ax_3d.set_xlabel(LABELS_PLOT[y_key], color=fg_col, labelpad=10, fontsize=14) 
        self.ax_3d.set_ylabel(LABELS_PLOT[x_key], color=fg_col, labelpad=10, fontsize=14) 
        self.ax_3d.set_zlabel(LABELS_PLOT[z_key], color=fg_col, labelpad=10, fontsize=14) 
        self.ax_3d.tick_params(colors=fg_col, labelsize=14) 
        self.ax_3d.set_zlim(0, 100)

        if y_key == "pH": self.ax_3d.set_xticks(np.arange(5.0, 9.1, 1.0))
        if x_key == "pH": self.ax_3d.set_yticks(np.arange(5.0, 9.1, 1.0))

        self.ax_contour.clear()
        self.ax_contour.set_facecolor(bg_col)
        contour = self.ax_contour.contourf(X_grid, Y_grid, Z_grid, levels=15, cmap='viridis')
        self.ax_contour.contour(X_grid, Y_grid, Z_grid, levels=15, colors=fg_col, linewidths=0.5, alpha=0.5)
        
        if not hasattr(self, 'cbar'):
            self.cbar = self.fig_3d.colorbar(contour, ax=self.ax_contour)
            for t in self.cbar.ax.get_yticklabels():
                t.set_color(fg_col)
        else:
            self.cbar.update_normal(contour)
            for t in self.cbar.ax.get_yticklabels():
                t.set_color(fg_col)

        self.ax_contour.set_xlabel(LABELS_PLOT[x_key], color=fg_col, fontsize=14) 
        self.ax_contour.set_ylabel(LABELS_PLOT[y_key], color=fg_col, fontsize=14) 
        self.ax_contour.tick_params(colors=fg_col, labelsize=14) 
        self.ax_contour.spines['bottom'].set_color(fg_col) 
        self.ax_contour.spines['left'].set_color(fg_col) 
        self.ax_contour.spines['top'].set_visible(False)
        self.ax_contour.spines['right'].set_visible(False)

        self.fig_3d.tight_layout()
        self.fig_3d.subplots_adjust(wspace=0.5)
        self.canvas_3d.draw()

if __name__ == "__main__":
    app = MembraneApp()
    app.mainloop()