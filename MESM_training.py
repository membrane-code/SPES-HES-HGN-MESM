from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor

try:
    import optuna
except Exception as exc:
    raise ImportError(
        "Optuna is required for the ESI-described MESM workflow. "
        "Install it using: pip install optuna"
    ) from exc


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


RANDOM_SEED = 42

INPUTS = ["Pressure", "Conc", "pH"]
OUTPUTS = ["Flux", "SMX", "TRM", "TET", "ERY"]
ACTIVATIONS = ["relu", "tanh", "logistic"]

DOMAIN = {
    "Pressure": (0.5, 3.5),
    "Conc": (2.0, 10.0),
    "pH": (5.0, 9.0),
}

DEFAULT_N_TRIALS = 25000

GUI_X_MEAN = {
    "Pressure": 2.0,
    "Conc": 6.0,
    "pH": 7.0,
}

GUI_X_STD = {
    "Pressure": 1.1767,
    "Conc": 3.1379,
    "pH": 1.5689,
}

GUI_Y_MEAN = {
    "Flux": 47.2377,
    "SMX": 56.7977,
    "TRM": 59.2708,
    "TET": 66.2700,
    "ERY": 40.3446,
}

GUI_Y_STD = {
    "Flux": 22.6403,
    "SMX": 24.9302,
    "TRM": 32.0292,
    "TET": 28.0572,
    "ERY": 24.7720,
}

GUI_META_PARAMS = {
    "Flux": [-0.068333285, 1.373323147, -0.104341967, -0.544899650],
    "SMX": [0.015567220, 1.335469746, 0.174309074, -0.720833601],
    "TRM": [0.047666143, -0.580855515, 1.129398511, 0.317550521],
    "TET": [-0.020643431, 0.414042676, 0.235800880, 0.461972832],
    "ERY": [-0.005308616, 1.083363941, 0.363772469, -0.679546371],
}

EMBEDDED_DATABASE = [
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
    [3.5, 10, 7, 78.13, 83.58, 92.33, 96.98, 82.13],
]

COLUMN_ALIASES = {
    "Run": [
        "Run",
        "Run #",
        "Run Number",
        "Experiment",
        "Experiment No",
        "#",
    ],
    "Pressure": [
        "Pressure",
        "Pressure (bar)",
        "P",
        "P (bar)",
        "Transmembrane pressure",
        "Transmembrane pressure (bar)",
    ],
    "Conc": [
        "Conc",
        "Concentration",
        "Concentration (mg/L)",
        "C0",
        "C_0",
        "C 0",
        "Ci",
        "C_i",
        "Initial concentration",
        "Initial antibiotic concentration",
        "Initial antibiotic concentration (mg/L)",
        "Feed concentration",
        "Feed concentration (mg/L)",
    ],
    "pH": [
        "pH",
        "PH",
        "Feed pH",
    ],
    "Flux": [
        "Flux",
        "Flux (LMH)",
        "J",
        "J (LMH)",
        "Permeate flux",
        "Permeate flux (LMH)",
    ],
    "SMX": [
        "SMX",
        "SMX Rejection",
        "SMX Rejection (%)",
        "RSMX",
        "R_SMX",
        "Sulfamethoxazole",
        "Sulfamethoxazole rejection",
    ],
    "TRM": [
        "TRM",
        "TRM Rejection",
        "TRM Rejection (%)",
        "RTRM",
        "R_TRM",
        "Trimethoprim",
        "Trimethoprim rejection",
    ],
    "TET": [
        "TET",
        "TET Rejection",
        "TET Rejection (%)",
        "RTET",
        "R_TET",
        "Tetracycline",
        "Tetracycline rejection",
    ],
    "ERY": [
        "ERY",
        "ERY Rejection",
        "ERY Rejection (%)",
        "RERY",
        "R_ERY",
        "Erythromycin",
        "Erythromycin rejection",
    ],
}


@dataclass(frozen=True)
class ScalingStats:
    x_mean: Dict[str, float]
    x_std: Dict[str, float]
    y_mean: Dict[str, float]
    y_std: Dict[str, float]
    source: str


@dataclass(frozen=True)
class TrainingSummary:
    random_seed: int
    n_samples: int
    inputs: List[str]
    outputs: List[str]
    domain: Dict[str, Tuple[float, float]]
    activations: List[str]
    n_trials_per_activation_per_outer_fold: int
    optuna_sampler: str
    optuna_pruner: str
    outer_cv: str
    inner_cv: str
    objective: str
    solver: str
    regularization: str
    standardization: str
    scaling_source: str
    selected_consensus_hyperparameters: Dict[str, Dict[str, object]]
    trainable_parameters_by_branch: Dict[str, int]
    total_trainable_parameters: int
    scaling: ScalingStats


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_column_name(name: str) -> str:
    text = str(name).strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def find_column(df: pd.DataFrame, canonical: str, required: bool = True) -> Optional[str]:
    normalized_lookup = {normalize_column_name(col): col for col in df.columns}

    for alias in COLUMN_ALIASES.get(canonical, []):
        key = normalize_column_name(alias)
        if key in normalized_lookup:
            return normalized_lookup[key]

    canonical_key = normalize_column_name(canonical)
    for col in df.columns:
        col_key = normalize_column_name(col)
        if canonical_key in col_key:
            return col

    if required:
        raise KeyError(
            f"Could not find a database column for '{canonical}'. "
            f"Available columns are: {list(df.columns)}"
        )

    return None


def load_database(path: Path, sheet_name: Optional[str] = None, use_embedded_if_missing: bool = False) -> pd.DataFrame:
    if not path.exists():
        if use_embedded_if_missing:
            df = pd.DataFrame(EMBEDDED_DATABASE, columns=INPUTS + OUTPUTS)
            df.insert(0, "Run", np.arange(1, len(df) + 1))
            return df
        raise FileNotFoundError(f"Database not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet_name or 0)
    elif suffix == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError("Database file must be .xlsx, .xls, or .csv.")

    run_col = find_column(raw, "Run", required=False)
    mapping = {name: find_column(raw, name, required=True) for name in INPUTS + OUTPUTS}

    columns_to_keep = []
    final_names = []

    if run_col is not None:
        columns_to_keep.append(run_col)
        final_names.append("Run")

    for name in INPUTS + OUTPUTS:
        columns_to_keep.append(mapping[name])
        final_names.append(name)

    df = raw[columns_to_keep].copy()
    df.columns = final_names

    for col in INPUTS + OUTPUTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=INPUTS + OUTPUTS).reset_index(drop=True)

    if "Run" not in df.columns:
        df.insert(0, "Run", np.arange(1, len(df) + 1))

    if df.empty:
        raise ValueError("No complete numeric BBD rows were found in the database.")

    return df


def validate_database(df: pd.DataFrame) -> List[str]:
    messages = []

    if len(df) != 13:
        messages.append(f"The ESI reports N = 13 BBD experiments, but {len(df)} rows were found.")

    for name, (low, high) in DOMAIN.items():
        observed_low = float(df[name].min())
        observed_high = float(df[name].max())

        if observed_low < low or observed_high > high:
            messages.append(
                f"{name} extends outside the ESI domain: observed {observed_low:g}-{observed_high:g}; "
                f"expected {low:g}-{high:g}."
            )

    return messages


def build_scaling_stats(df: pd.DataFrame, scaling_source: str) -> ScalingStats:
    if scaling_source == "gui":
        return ScalingStats(
            x_mean={name: float(GUI_X_MEAN[name]) for name in INPUTS},
            x_std={name: float(GUI_X_STD[name]) for name in INPUTS},
            y_mean={name: float(GUI_Y_MEAN[name]) for name in OUTPUTS},
            y_std={name: float(GUI_Y_STD[name]) for name in OUTPUTS},
            source="gui",
        )

    if scaling_source == "database":
        x = df[INPUTS].astype(float)
        y = df[OUTPUTS].astype(float)

        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0, ddof=0)
        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0, ddof=0)

        if (x_std <= 0).any() or (y_std <= 0).any():
            raise ValueError("All inputs and outputs must have non-zero standard deviation.")

        return ScalingStats(
            x_mean={name: float(x_mean[name]) for name in INPUTS},
            x_std={name: float(x_std[name]) for name in INPUTS},
            y_mean={name: float(y_mean[name]) for name in OUTPUTS},
            y_std={name: float(y_std[name]) for name in OUTPUTS},
            source="database",
        )

    raise ValueError("scaling_source must be either 'gui' or 'database'.")


def standardize(df: pd.DataFrame, stats: ScalingStats) -> Tuple[np.ndarray, np.ndarray]:
    x = df[INPUTS].astype(float).to_numpy()
    y = df[OUTPUTS].astype(float).to_numpy()

    x_mean = np.array([stats.x_mean[name] for name in INPUTS], dtype=float).reshape(1, -1)
    x_std = np.array([stats.x_std[name] for name in INPUTS], dtype=float).reshape(1, -1)
    y_mean = np.array([stats.y_mean[name] for name in OUTPUTS], dtype=float).reshape(1, -1)
    y_std = np.array([stats.y_std[name] for name in OUTPUTS], dtype=float).reshape(1, -1)

    if np.any(x_std <= 0) or np.any(y_std <= 0):
        raise ValueError("Scaling standard deviations must be positive.")

    x_z = (x - x_mean) / x_std
    y_z = (y - y_mean) / y_std

    return x_z, y_z


def compare_database_to_gui_stats(df: pd.DataFrame) -> pd.DataFrame:
    x = df[INPUTS].astype(float)
    y = df[OUTPUTS].astype(float)

    calculated = {
        "X_mean": x.mean(axis=0),
        "X_std_population": x.std(axis=0, ddof=0),
        "Y_mean": y.mean(axis=0),
        "Y_std_population": y.std(axis=0, ddof=0),
    }

    rows = []

    for name in INPUTS:
        rows.append(
            {
                "Variable": name,
                "Type": "Input mean",
                "Calculated": float(calculated["X_mean"][name]),
                "GUI_value": float(GUI_X_MEAN[name]),
                "Absolute_difference": float(abs(calculated["X_mean"][name] - GUI_X_MEAN[name])),
            }
        )
        rows.append(
            {
                "Variable": name,
                "Type": "Input std",
                "Calculated": float(calculated["X_std_population"][name]),
                "GUI_value": float(GUI_X_STD[name]),
                "Absolute_difference": float(abs(calculated["X_std_population"][name] - GUI_X_STD[name])),
            }
        )

    for name in OUTPUTS:
        rows.append(
            {
                "Variable": name,
                "Type": "Output mean",
                "Calculated": float(calculated["Y_mean"][name]),
                "GUI_value": float(GUI_Y_MEAN[name]),
                "Absolute_difference": float(abs(calculated["Y_mean"][name] - GUI_Y_MEAN[name])),
            }
        )
        rows.append(
            {
                "Variable": name,
                "Type": "Output std",
                "Calculated": float(calculated["Y_std_population"][name]),
                "GUI_value": float(GUI_Y_STD[name]),
                "Absolute_difference": float(abs(calculated["Y_std_population"][name] - GUI_Y_STD[name])),
            }
        )

    return pd.DataFrame(rows)


def parameter_count(n_input: int, hidden_layer_sizes: Sequence[int], n_output: int) -> int:
    layer_sizes = [int(n_input)] + [int(v) for v in hidden_layer_sizes] + [int(n_output)]
    total = 0

    for current_size, next_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += current_size * next_size
        total += next_size

    return int(total)


def make_mlp(activation: str, params: Mapping[str, object], seed: int) -> MLPRegressor:
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}")

    return MLPRegressor(
        hidden_layer_sizes=tuple(int(v) for v in params["hidden_layer_sizes"]),
        activation=activation,
        solver="lbfgs",
        alpha=float(params["alpha"]),
        max_iter=int(params["max_iter"]),
        max_fun=max(2000, int(params["max_iter"]) * 3),
        random_state=int(seed),
        tol=1.0e-7,
        early_stopping=False,
    )


def trial_to_params(trial: optuna.Trial, activation: str) -> Dict[str, object]:
    n_layers = trial.suggest_int(f"{activation}_n_layers", 2, 6)

    hidden_layer_sizes = []
    for layer_id in range(n_layers):
        hidden_layer_sizes.append(
            trial.suggest_int(
                f"{activation}_n_units_l{layer_id + 1}",
                4,
                32,
            )
        )

    alpha = trial.suggest_float(
        f"{activation}_alpha",
        1.0e-6,
        1.0e-1,
        log=True,
    )

    max_iter = trial.suggest_int(
        f"{activation}_max_iter",
        250,
        10000,
        step=250,
    )

    return {
        "hidden_layer_sizes": tuple(int(v) for v in hidden_layer_sizes),
        "alpha": float(alpha),
        "max_iter": int(max_iter),
    }


def safe_mean_multioutput_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []

    for output_id in range(y_true.shape[1]):
        truth = y_true[:, output_id]
        pred = y_pred[:, output_id]

        if len(truth) < 2:
            scores.append(0.0)
            continue

        if np.allclose(truth, truth[0]):
            scores.append(0.0)
            continue

        score = r2_score(truth, pred)

        if np.isfinite(score):
            scores.append(float(score))
        else:
            scores.append(0.0)

    return float(np.mean(scores))


def inner_loo_objective_score(
    x_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    activation: str,
    params: Mapping[str, object],
    seed: int,
    trial: Optional[optuna.Trial] = None,
) -> float:
    inner = LeaveOneOut()
    y_oof = np.zeros_like(y_train_outer, dtype=float)
    completed_indices: List[int] = []

    for inner_fold_id, (inner_train_idx, inner_valid_idx) in enumerate(inner.split(x_train_outer)):
        try:
            model = make_mlp(
                activation=activation,
                params=params,
                seed=seed + inner_fold_id,
            )

            model.fit(
                x_train_outer[inner_train_idx],
                y_train_outer[inner_train_idx],
            )

            pred = np.asarray(model.predict(x_train_outer[inner_valid_idx]), dtype=float)

        except Exception:
            return -1.0e12

        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        y_oof[inner_valid_idx, :] = pred
        completed_indices.extend(inner_valid_idx.tolist())

        if trial is not None and len(completed_indices) >= 2:
            partial_idx = np.array(completed_indices, dtype=int)
            partial_score = safe_mean_multioutput_r2(
                y_train_outer[partial_idx],
                y_oof[partial_idx],
            )

            trial.report(partial_score, step=inner_fold_id)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return safe_mean_multioutput_r2(y_train_outer, y_oof)


def optimize_one_activation_one_outer_fold(
    x_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    activation: str,
    outer_fold_id: int,
    n_trials: int,
    seed: int,
    storage: Optional[str],
    optuna_dir: Path,
    optuna_n_jobs: int,
) -> Dict[str, object]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study_name = f"mesm_{activation}_outer_fold_{outer_fold_id + 1:02d}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = trial_to_params(trial, activation)
        return inner_loo_objective_score(
            x_train_outer=x_train_outer,
            y_train_outer=y_train_outer,
            activation=activation,
            params=params,
            seed=seed + trial.number,
            trial=trial,
        )

    existing_trials = len(study.trials)
    remaining_trials = max(0, int(n_trials) - existing_trials)

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=max(1, int(optuna_n_jobs)),
            show_progress_bar=False,
            gc_after_trial=True,
        )

    if study.best_trial is None:
        raise RuntimeError(f"No successful Optuna trial was obtained for {study_name}.")

    best_params = trial_to_params(study.best_trial, activation)

    record = {
        "activation": activation,
        "outer_fold": int(outer_fold_id + 1),
        "held_out_run_index_1_based": int(outer_fold_id + 1),
        "best_value_mean_multioutput_r2": float(study.best_value),
        "hidden_layer_sizes": tuple(int(v) for v in best_params["hidden_layer_sizes"]),
        "alpha": float(best_params["alpha"]),
        "max_iter": int(best_params["max_iter"]),
        "n_trials_in_study": int(len(study.trials)),
        "n_trials_requested": int(n_trials),
        "parameter_count": int(
            parameter_count(
                n_input=len(INPUTS),
                hidden_layer_sizes=best_params["hidden_layer_sizes"],
                n_output=len(OUTPUTS),
            )
        ),
    }

    optuna_dir.mkdir(parents=True, exist_ok=True)
    with open(optuna_dir / f"{study_name}_best_params.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "activation": record["activation"],
                "outer_fold": record["outer_fold"],
                "held_out_run_index_1_based": record["held_out_run_index_1_based"],
                "best_value_mean_multioutput_r2": record["best_value_mean_multioutput_r2"],
                "hidden_layer_sizes": list(record["hidden_layer_sizes"]),
                "alpha": record["alpha"],
                "max_iter": record["max_iter"],
                "n_trials_in_study": record["n_trials_in_study"],
                "n_trials_requested": record["n_trials_requested"],
                "parameter_count": record["parameter_count"],
            },
            handle,
            indent=2,
        )

    return record


def nested_loo_base_predictions(
    x_z: np.ndarray,
    y_z: np.ndarray,
    n_trials: int,
    seed: int,
    storage: Optional[str],
    optuna_dir: Path,
    optuna_n_jobs: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    n_samples = x_z.shape[0]
    n_outputs = y_z.shape[1]

    base_oof = np.zeros((len(ACTIVATIONS), n_samples, n_outputs), dtype=float)
    records = []

    outer = LeaveOneOut()

    for outer_fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer.split(x_z)):
        x_train_outer = x_z[outer_train_idx]
        y_train_outer = y_z[outer_train_idx]

        for activation_id, activation in enumerate(ACTIVATIONS):
            best_record = optimize_one_activation_one_outer_fold(
                x_train_outer=x_train_outer,
                y_train_outer=y_train_outer,
                activation=activation,
                outer_fold_id=outer_fold_id,
                n_trials=n_trials,
                seed=seed + 100000 * activation_id + 1000 * outer_fold_id,
                storage=storage,
                optuna_dir=optuna_dir,
                optuna_n_jobs=optuna_n_jobs,
            )

            records.append(best_record)

            model = make_mlp(
                activation=activation,
                params={
                    "hidden_layer_sizes": best_record["hidden_layer_sizes"],
                    "alpha": best_record["alpha"],
                    "max_iter": best_record["max_iter"],
                },
                seed=seed + 100000 * activation_id + 1000 * outer_fold_id + 777,
            )

            model.fit(x_train_outer, y_train_outer)

            pred = np.asarray(model.predict(x_z[outer_test_idx]), dtype=float)
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)

            base_oof[activation_id, outer_test_idx, :] = pred

    return base_oof, pd.DataFrame(records)


def consensus_hyperparameters(records_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    consensus: Dict[str, Dict[str, object]] = {}

    for activation in ACTIVATIONS:
        subset = records_df[records_df["activation"] == activation].copy()

        if subset.empty:
            raise ValueError(f"No selected hyperparameter records found for {activation}.")

        hidden_candidates = [tuple(v) for v in subset["hidden_layer_sizes"].tolist()]
        hidden_counts = Counter(hidden_candidates)
        max_count = max(hidden_counts.values())
        tied_hidden = [h for h, count in hidden_counts.items() if count == max_count]

        if len(tied_hidden) == 1:
            hidden_mode = tied_hidden[0]
        else:
            best_score_by_hidden = {}
            for hidden in tied_hidden:
                local = subset[subset["hidden_layer_sizes"].apply(tuple) == hidden]
                best_score_by_hidden[hidden] = float(local["best_value_mean_multioutput_r2"].median())
            hidden_mode = max(best_score_by_hidden, key=best_score_by_hidden.get)

        alpha = float(np.median(subset["alpha"].astype(float).to_numpy()))
        max_iter = int(np.median(subset["max_iter"].astype(int).to_numpy()))

        consensus[activation] = {
            "hidden_layer_sizes": tuple(int(v) for v in hidden_mode),
            "alpha": alpha,
            "max_iter": max_iter,
        }

    return consensus


def train_final_base_models(
    x_z: np.ndarray,
    y_z: np.ndarray,
    configs: Mapping[str, Mapping[str, object]],
    seed: int,
) -> Dict[str, MLPRegressor]:
    models: Dict[str, MLPRegressor] = {}

    for activation_id, activation in enumerate(ACTIVATIONS):
        model = make_mlp(
            activation=activation,
            params=configs[activation],
            seed=seed + 1000 * activation_id + 999,
        )

        model.fit(x_z, y_z)
        models[activation] = model

    return models


def base_predictions(
    models: Mapping[str, MLPRegressor],
    x_z: np.ndarray,
) -> np.ndarray:
    predictions = []

    for activation in ACTIVATIONS:
        pred = np.asarray(models[activation].predict(x_z), dtype=float)

        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        predictions.append(pred)

    return np.stack(predictions, axis=0)


def fit_meta_layer(base_oof: np.ndarray, y_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_outputs = y_z.shape[1]
    n_models = base_oof.shape[0]

    weights = np.zeros((n_outputs, n_models), dtype=float)
    intercepts = np.zeros(n_outputs, dtype=float)

    for output_id in range(n_outputs):
        x_meta = base_oof[:, :, output_id].T
        y_meta = y_z[:, output_id]

        reg = LinearRegression()
        reg.fit(x_meta, y_meta)

        intercepts[output_id] = float(reg.intercept_)
        weights[output_id, :] = np.asarray(reg.coef_, dtype=float)

    return weights, intercepts


def gui_fallback_meta_layer() -> Tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((len(OUTPUTS), len(ACTIVATIONS)), dtype=float)
    intercepts = np.zeros(len(OUTPUTS), dtype=float)

    for output_id, output_name in enumerate(OUTPUTS):
        params = GUI_META_PARAMS[output_name]
        intercepts[output_id] = float(params[0])
        weights[output_id, :] = np.asarray(params[1:], dtype=float)

    return weights, intercepts


def ensemble_from_base(
    base_preds: np.ndarray,
    weights: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    return np.einsum("ano,oa->no", base_preds, weights) + intercepts.reshape(1, -1)


def uncertainty_from_base(
    base_preds: np.ndarray,
    weights: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    abs_weights = np.abs(weights)
    denom = abs_weights.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)

    normalized_weights = abs_weights / denom
    weighted_mean_z = np.einsum("ano,oa->no", base_preds, normalized_weights)

    diffs = base_preds - weighted_mean_z[None, :, :]
    var_z = np.einsum("ano,oa->no", diffs * diffs, normalized_weights)

    return np.sqrt(np.maximum(var_z, 0.0)) * y_std.reshape(1, -1)


def inverse_y(y_z: np.ndarray, stats: ScalingStats) -> np.ndarray:
    y_mean = np.array([stats.y_mean[name] for name in OUTPUTS], dtype=float).reshape(1, -1)
    y_std = np.array([stats.y_std[name] for name in OUTPUTS], dtype=float).reshape(1, -1)

    return y_z * y_std + y_mean


def clip_physical_outputs(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    out[:, 0] = np.maximum(out[:, 0], 0.0)
    out[:, 1:] = np.clip(out[:, 1:], 0.0, 100.0)

    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    evaluation_label: str,
) -> pd.DataFrame:
    rows = []

    for output_id, output_name in enumerate(OUTPUTS):
        truth = y_true[:, output_id]
        pred = y_pred[:, output_id]

        rmse = float(math.sqrt(mean_squared_error(truth, pred)))
        mae = float(mean_absolute_error(truth, pred))
        r2 = float(r2_score(truth, pred))

        denom = np.where(np.abs(truth) < 1.0e-12, np.nan, np.abs(truth))
        aard = float(np.nanmean(np.abs((truth - pred) / denom)) * 100.0)

        rows.append(
            {
                "Evaluation": evaluation_label,
                "Output": output_name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "AARD_percent": aard,
            }
        )

    return pd.DataFrame(rows)


def write_scaling_json(stats: ScalingStats, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, indent=2)


def write_meta_weights_csv(
    weights: np.ndarray,
    intercepts: np.ndarray,
    out_path: Path,
) -> None:
    rows = []

    for output_id, output_name in enumerate(OUTPUTS):
        row = {
            "Output": output_name,
            "Intercept": float(intercepts[output_id]),
        }

        for activation_id, activation in enumerate(ACTIVATIONS):
            row[f"Weight_{activation}"] = float(weights[output_id, activation_id])

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def make_oof_base_frame(
    df: pd.DataFrame,
    base_oof_z: np.ndarray,
    stats: ScalingStats,
) -> pd.DataFrame:
    rows = []

    for sample_id in range(len(df)):
        base_physical = clip_physical_outputs(
            inverse_y(base_oof_z[:, sample_id, :], stats)
        )

        row = {
            "Run": df.loc[sample_id, "Run"],
            "Pressure": df.loc[sample_id, "Pressure"],
            "Conc": df.loc[sample_id, "Conc"],
            "pH": df.loc[sample_id, "pH"],
        }

        for activation_id, activation in enumerate(ACTIVATIONS):
            for output_id, output_name in enumerate(OUTPUTS):
                row[f"OOF_{activation}_{output_name}"] = float(base_physical[activation_id, output_id])

        rows.append(row)

    return pd.DataFrame(rows)


def make_prediction_frame(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    out = df[["Run"] + INPUTS + OUTPUTS].copy()

    for output_id, output_name in enumerate(OUTPUTS):
        out[f"{prefix}_{output_name}"] = np.round(y_pred[:, output_id], 6)
        out[f"{prefix}_Uncertainty_{output_name}"] = np.round(uncertainty[:, output_id], 6)

    return out


def predict_ensemble_physical(
    models: Mapping[str, MLPRegressor],
    weights: np.ndarray,
    intercepts: np.ndarray,
    x_raw: np.ndarray,
    stats: ScalingStats,
) -> Tuple[np.ndarray, np.ndarray]:
    x_mean = np.array([stats.x_mean[name] for name in INPUTS], dtype=float).reshape(1, -1)
    x_std = np.array([stats.x_std[name] for name in INPUTS], dtype=float).reshape(1, -1)

    x_z = (np.asarray(x_raw, dtype=float) - x_mean) / x_std

    base_z = base_predictions(models, x_z)
    ens_z = ensemble_from_base(base_z, weights, intercepts)

    pred = clip_physical_outputs(inverse_y(ens_z, stats))

    y_std = np.array([stats.y_std[name] for name in OUTPUTS], dtype=float)
    uncertainty = uncertainty_from_base(base_z, weights, y_std)

    return pred, uncertainty


def write_response_grid(
    models: Mapping[str, MLPRegressor],
    weights: np.ndarray,
    intercepts: np.ndarray,
    stats: ScalingStats,
    out_path: Path,
    resolution: int,
) -> None:
    pressure = np.linspace(DOMAIN["Pressure"][0], DOMAIN["Pressure"][1], resolution)
    conc = np.linspace(DOMAIN["Conc"][0], DOMAIN["Conc"][1], resolution)
    ph = np.linspace(DOMAIN["pH"][0], DOMAIN["pH"][1], resolution)

    p_grid, c_grid, ph_grid = np.meshgrid(pressure, conc, ph, indexing="ij")
    x_raw = np.column_stack([p_grid.ravel(), c_grid.ravel(), ph_grid.ravel()])

    pred, uncertainty = predict_ensemble_physical(
        models=models,
        weights=weights,
        intercepts=intercepts,
        x_raw=x_raw,
        stats=stats,
    )

    grid = pd.DataFrame(x_raw, columns=INPUTS)

    for output_id, output_name in enumerate(OUTPUTS):
        grid[f"Pred_{output_name}"] = np.round(pred[:, output_id], 6)
        grid[f"Uncertainty_{output_name}"] = np.round(uncertainty[:, output_id], 6)

    grid.to_csv(out_path, index=False)


def compute_permutation_importance(
    models: Mapping[str, MLPRegressor],
    weights: np.ndarray,
    intercepts: np.ndarray,
    x_z: np.ndarray,
    y_z: np.ndarray,
    out_path: Path,
    seed: int,
    n_repeats: int = 200,
) -> None:
    rng = np.random.default_rng(seed)

    baseline_base = base_predictions(models, x_z)
    baseline_pred = ensemble_from_base(baseline_base, weights, intercepts)

    rows = []

    for output_id, output_name in enumerate(OUTPUTS):
        baseline_mse = mean_squared_error(y_z[:, output_id], baseline_pred[:, output_id])

        for feature_id, feature_name in enumerate(INPUTS):
            increases = []

            for _ in range(n_repeats):
                x_perm = x_z.copy()
                x_perm[:, feature_id] = rng.permutation(x_perm[:, feature_id])

                perm_base = base_predictions(models, x_perm)
                perm_pred = ensemble_from_base(perm_base, weights, intercepts)
                perm_mse = mean_squared_error(y_z[:, output_id], perm_pred[:, output_id])

                increases.append(float(perm_mse - baseline_mse))

            rows.append(
                {
                    "Output": output_name,
                    "Feature": feature_name,
                    "Permutation_MSE_Increase_mean": float(np.mean(increases)),
                    "Permutation_MSE_Increase_std": float(np.std(increases, ddof=0)),
                }
            )

    pd.DataFrame(rows).to_csv(out_path, index=False)


def unpack_shap_values(shap_values: object, n_outputs: int) -> List[np.ndarray]:
    if isinstance(shap_values, list):
        arrays = [np.asarray(item, dtype=float) for item in shap_values]
        if len(arrays) != n_outputs:
            raise ValueError(f"Expected {n_outputs} SHAP arrays, received {len(arrays)}.")
        return arrays

    arr = np.asarray(shap_values, dtype=float)

    if arr.ndim == 2 and n_outputs == 1:
        return [arr]

    if arr.ndim == 3:
        if arr.shape[2] == n_outputs:
            return [arr[:, :, output_id] for output_id in range(n_outputs)]

        if arr.shape[0] == n_outputs:
            return [arr[output_id, :, :] for output_id in range(n_outputs)]

    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def compute_shap_importance(
    models: Mapping[str, MLPRegressor],
    weights: np.ndarray,
    x_z: np.ndarray,
    importance_out_path: Path,
    values_out_path: Path,
    nsamples: object = "auto",
) -> None:
    try:
        import shap
    except Exception as exc:
        raise ImportError("SHAP is required for --compute-shap. Install it using: pip install shap") from exc

    background = x_z
    n_samples = x_z.shape[0]
    n_features = x_z.shape[1]
    n_outputs = len(OUTPUTS)
    n_models = len(ACTIVATIONS)

    base_shap = np.zeros(
        (n_models, n_outputs, n_samples, n_features),
        dtype=float,
    )

    for model_id, activation in enumerate(ACTIVATIONS):
        model = models[activation]

        def base_model_function(x_input: np.ndarray) -> np.ndarray:
            pred = np.asarray(model.predict(np.asarray(x_input, dtype=float)), dtype=float)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            return pred

        explainer = shap.KernelExplainer(base_model_function, background)
        shap_values = explainer.shap_values(x_z, nsamples=nsamples)
        shap_arrays = unpack_shap_values(shap_values, n_outputs=n_outputs)

        for output_id in range(n_outputs):
            arr = np.asarray(shap_arrays[output_id], dtype=float)

            if arr.shape != (n_samples, n_features):
                raise ValueError(
                    f"Unexpected SHAP array shape for {activation}/{OUTPUTS[output_id]}: {arr.shape}"
                )

            base_shap[model_id, output_id, :, :] = arr

    ensemble_shap = np.zeros(
        (n_outputs, n_samples, n_features),
        dtype=float,
    )

    for output_id in range(n_outputs):
        for model_id in range(n_models):
            ensemble_shap[output_id, :, :] += (
                float(weights[output_id, model_id]) * base_shap[model_id, output_id, :, :]
            )

    importance_rows = []
    value_rows = []

    for output_id, output_name in enumerate(OUTPUTS):
        mean_abs = np.mean(np.abs(ensemble_shap[output_id, :, :]), axis=0)
        total = float(np.sum(mean_abs))

        for feature_id, feature_name in enumerate(INPUTS):
            fraction = 0.0 if total == 0.0 else float(mean_abs[feature_id] / total)

            importance_rows.append(
                {
                    "Output": output_name,
                    "Feature": feature_name,
                    "MeanAbsSHAP_z": float(mean_abs[feature_id]),
                    "Fraction": fraction,
                    "Percent": float(100.0 * fraction),
                }
            )

        for sample_id in range(n_samples):
            for feature_id, feature_name in enumerate(INPUTS):
                value_rows.append(
                    {
                        "Sample_index_1_based": int(sample_id + 1),
                        "Output": output_name,
                        "Feature": feature_name,
                        "Ensemble_SHAP_z": float(ensemble_shap[output_id, sample_id, feature_id]),
                    }
                )

    pd.DataFrame(importance_rows).to_csv(importance_out_path, index=False)
    pd.DataFrame(value_rows).to_csv(values_out_path, index=False)


def save_pickles(
    models: Mapping[str, MLPRegressor],
    weights: np.ndarray,
    intercepts: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for activation in ACTIVATIONS:
        with open(out_dir / f"{activation}_model.pkl", "wb") as handle:
            pickle.dump(models[activation], handle)

    with open(out_dir / "meta_layer_model.pkl", "wb") as handle:
        pickle.dump(
            {
                "weights": np.asarray(weights, dtype=float),
                "intercepts": np.asarray(intercepts, dtype=float),
                "activations": list(ACTIVATIONS),
                "outputs": list(OUTPUTS),
            },
            handle,
        )


def write_summary_json(summary: TrainingSummary, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)


def is_extrapolation(pressure: float, conc: float, ph: float) -> bool:
    values = {
        "Pressure": pressure,
        "Conc": conc,
        "pH": ph,
    }

    for key, value in values.items():
        low, high = DOMAIN[key]
        if value < low or value > high:
            return True

    return False


def load_exported_prediction_system(
    models_dir: Path,
    scaling_path: Path,
) -> Tuple[Dict[str, MLPRegressor], np.ndarray, np.ndarray, ScalingStats]:
    models: Dict[str, MLPRegressor] = {}

    for activation in ACTIVATIONS:
        with open(models_dir / f"{activation}_model.pkl", "rb") as handle:
            models[activation] = pickle.load(handle)

    with open(models_dir / "meta_layer_model.pkl", "rb") as handle:
        meta = pickle.load(handle)

    with open(scaling_path, "r", encoding="utf-8") as handle:
        scaling_payload = json.load(handle)

    stats = ScalingStats(
        x_mean={k: float(v) for k, v in scaling_payload["x_mean"].items()},
        x_std={k: float(v) for k, v in scaling_payload["x_std"].items()},
        y_mean={k: float(v) for k, v in scaling_payload["y_mean"].items()},
        y_std={k: float(v) for k, v in scaling_payload["y_std"].items()},
        source=str(scaling_payload.get("source", "unknown")),
    )

    return (
        models,
        np.asarray(meta["weights"], dtype=float),
        np.asarray(meta["intercepts"], dtype=float),
        stats,
    )


def predict_single_from_exported_files(
    models_dir: Path,
    scaling_path: Path,
    pressure: float,
    conc: float,
    ph: float,
) -> Dict[str, object]:
    models, weights, intercepts, stats = load_exported_prediction_system(
        models_dir=models_dir,
        scaling_path=scaling_path,
    )

    x_raw = np.array([[pressure, conc, ph]], dtype=float)

    pred, uncertainty = predict_ensemble_physical(
        models=models,
        weights=weights,
        intercepts=intercepts,
        x_raw=x_raw,
        stats=stats,
    )

    result: Dict[str, object] = {
        "Pressure": float(pressure),
        "Conc": float(conc),
        "pH": float(ph),
        "extrapolation": bool(is_extrapolation(pressure, conc, ph)),
    }

    for output_id, output_name in enumerate(OUTPUTS):
        result[output_name] = float(pred[0, output_id])
        result[f"{output_name}_uncertainty"] = float(uncertainty[0, output_id])

    return result


def format_hidden_for_csv(value: object) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        return json.dumps([int(v) for v in value])

    return json.dumps(value)


def write_selected_records_csv(records_df: pd.DataFrame, out_path: Path) -> None:
    records_for_csv = records_df.copy()

    if "hidden_layer_sizes" in records_for_csv.columns:
        records_for_csv["hidden_layer_sizes"] = records_for_csv["hidden_layer_sizes"].apply(format_hidden_for_csv)

    records_for_csv.to_csv(out_path, index=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, interpret, and export the manuscript-consistent MESM ensemble."
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=Path("Database.xlsx"),
        help="Path to Database.xlsx, .xls, or .csv.",
    )

    parser.add_argument(
        "--sheet-name",
        type=str,
        default=None,
        help="Excel sheet name. If omitted, the first sheet is used.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MESM_outputs"),
        help="Output directory.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Optuna inner trials per activation per outer LOOCV fold. Default is 25,000.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI. If omitted, SQLite storage is created inside output-dir.",
    )

    parser.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna jobs within each study. Use 1 for strict reproducibility.",
    )

    parser.add_argument(
        "--scaling-source",
        choices=["gui", "database"],
        default="gui",
        help="Use GUI scaling constants or recompute population z-score statistics from Database.xlsx.",
    )

    parser.add_argument(
        "--use-embedded-data-if-missing",
        action="store_true",
        help="Use the embedded 13-run BBD table only if Database.xlsx is not found.",
    )

    parser.add_argument(
        "--use-gui-meta",
        action="store_true",
        help="Use the GUI-embedded meta-layer coefficients instead of fitting OLS on OOF base predictions.",
    )

    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=41,
        help="Resolution per input axis for exported response grid.",
    )

    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=200,
        help="Number of repeats for permutation importance.",
    )

    parser.add_argument(
        "--compute-shap",
        action="store_true",
        help="Compute Kernel SHAP tables after final model training.",
    )

    parser.add_argument(
        "--shap-nsamples",
        default="auto",
        help="Kernel SHAP nsamples argument. Use auto or an integer such as 200.",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Print one prediction after training using --pressure, --conc, and --ph.",
    )

    parser.add_argument(
        "--pressure",
        type=float,
        default=2.0,
        help="Pressure for optional single prediction.",
    )

    parser.add_argument(
        "--conc",
        type=float,
        default=6.0,
        help="Concentration for optional single prediction.",
    )

    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH for optional single prediction.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    set_global_seed(args.seed)

    out_dir = args.output_dir
    models_dir = out_dir / "models"
    tables_dir = out_dir / "tables"
    optuna_dir = out_dir / "optuna_best_params"

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage
    if storage is None:
        storage_path = out_dir / "optuna_studies.db"
        storage = f"sqlite:///{storage_path.as_posix()}"

    df = load_database(
        path=args.database,
        sheet_name=args.sheet_name,
        use_embedded_if_missing=bool(args.use_embedded_data_if_missing),
    )

    validation_messages = validate_database(df)

    if validation_messages:
        for msg in validation_messages:
            print("WARNING:", msg)

    df.to_csv(tables_dir / "mesm_database_cleaned.csv", index=False)

    gui_stats_check = compare_database_to_gui_stats(df)
    gui_stats_check.to_csv(tables_dir / "mesm_database_vs_gui_scaling_check.csv", index=False)

    stats = build_scaling_stats(df, scaling_source=args.scaling_source)
    x_z, y_z = standardize(df, stats)
    y_true = df[OUTPUTS].to_numpy(dtype=float)

    if len(df) != 13:
        raise ValueError(
            f"The ESI describes a 13-run BBD dataset, but {len(df)} rows were found."
        )

    print(f"Loaded {len(df)} BBD rows from {args.database}.")
    print(f"Scaling source: {args.scaling_source}.")
    print(f"Using {args.n_trials} Optuna inner trials per activation per outer LOOCV fold.")
    print("Objective: maximize mean multi-output R2 across Flux, SMX, TRM, TET, and ERY.")
    print(f"Optuna storage: {storage}")

    base_oof_z, selected_records = nested_loo_base_predictions(
        x_z=x_z,
        y_z=y_z,
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        storage=storage,
        optuna_dir=optuna_dir,
        optuna_n_jobs=int(args.optuna_n_jobs),
    )

    write_selected_records_csv(
        records_df=selected_records,
        out_path=tables_dir / "mesm_selected_hyperparameters.csv",
    )

    consensus_configs = consensus_hyperparameters(selected_records)

    if args.use_gui_meta:
        weights, intercepts = gui_fallback_meta_layer()
        meta_label = "GUI_fallback_meta"
    else:
        weights, intercepts = fit_meta_layer(base_oof_z, y_z)
        meta_label = "OLS_on_outer_OOF_base_predictions"

    oof_ens_z = ensemble_from_base(base_oof_z, weights, intercepts)
    oof_ens = clip_physical_outputs(inverse_y(oof_ens_z, stats))

    y_std = np.array([stats.y_std[name] for name in OUTPUTS], dtype=float)
    oof_uncertainty = uncertainty_from_base(base_oof_z, weights, y_std)

    final_models = train_final_base_models(
        x_z=x_z,
        y_z=y_z,
        configs=consensus_configs,
        seed=int(args.seed),
    )

    full_base_z = base_predictions(final_models, x_z)
    full_ens_z = ensemble_from_base(full_base_z, weights, intercepts)
    full_ens = clip_physical_outputs(inverse_y(full_ens_z, stats))
    full_uncertainty = uncertainty_from_base(full_base_z, weights, y_std)

    metrics_oof = compute_metrics(
        y_true=y_true,
        y_pred=oof_ens,
        evaluation_label=f"Nested_LOOCV_OOF_{meta_label}",
    )

    metrics_full = compute_metrics(
        y_true=y_true,
        y_pred=full_ens,
        evaluation_label=f"Full_fit_consensus_{meta_label}",
    )

    metrics = pd.concat([metrics_oof, metrics_full], ignore_index=True)
    metrics.to_csv(tables_dir / "mesm_metrics.csv", index=False)

    make_oof_base_frame(
        df=df,
        base_oof_z=base_oof_z,
        stats=stats,
    ).to_csv(tables_dir / "mesm_outer_oof_base_predictions.csv", index=False)

    make_prediction_frame(
        df=df,
        y_pred=oof_ens,
        uncertainty=oof_uncertainty,
        prefix="OOF_Pred",
    ).to_csv(tables_dir / "mesm_outer_oof_ensemble_predictions.csv", index=False)

    make_prediction_frame(
        df=df,
        y_pred=full_ens,
        uncertainty=full_uncertainty,
        prefix="FullFit_Pred",
    ).to_csv(tables_dir / "mesm_full_fit_predictions.csv", index=False)

    write_meta_weights_csv(
        weights=weights,
        intercepts=intercepts,
        out_path=tables_dir / "mesm_meta_weights.csv",
    )

    write_scaling_json(
        stats=stats,
        out_path=tables_dir / "mesm_scaling_stats.json",
    )

    write_response_grid(
        models=final_models,
        weights=weights,
        intercepts=intercepts,
        stats=stats,
        out_path=tables_dir / "mesm_response_grid.csv",
        resolution=max(3, int(args.grid_resolution)),
    )

    compute_permutation_importance(
        models=final_models,
        weights=weights,
        intercepts=intercepts,
        x_z=x_z,
        y_z=y_z,
        out_path=tables_dir / "mesm_permutation_importance.csv",
        seed=int(args.seed),
        n_repeats=max(1, int(args.permutation_repeats)),
    )

    if args.compute_shap:
        shap_nsamples = args.shap_nsamples
        if isinstance(shap_nsamples, str) and shap_nsamples.lower() != "auto":
            try:
                shap_nsamples = int(shap_nsamples)
            except ValueError:
                raise ValueError("--shap-nsamples must be 'auto' or an integer.")

        compute_shap_importance(
            models=final_models,
            weights=weights,
            x_z=x_z,
            importance_out_path=tables_dir / "mesm_shap_importance.csv",
            values_out_path=tables_dir / "mesm_shap_values_long.csv",
            nsamples=shap_nsamples,
        )

    save_pickles(
        models=final_models,
        weights=weights,
        intercepts=intercepts,
        out_dir=models_dir,
    )

    param_counts = {
        activation: parameter_count(
            n_input=len(INPUTS),
            hidden_layer_sizes=consensus_configs[activation]["hidden_layer_sizes"],
            n_output=len(OUTPUTS),
        )
        for activation in ACTIVATIONS
    }

    summary = TrainingSummary(
        random_seed=int(args.seed),
        n_samples=int(len(df)),
        inputs=list(INPUTS),
        outputs=list(OUTPUTS),
        domain={key: tuple(float(v) for v in vals) for key, vals in DOMAIN.items()},
        activations=list(ACTIVATIONS),
        n_trials_per_activation_per_outer_fold=int(args.n_trials),
        optuna_sampler="TPESampler",
        optuna_pruner="MedianPruner(n_warmup_steps=5)",
        outer_cv="LeaveOneOut(N=13)",
        inner_cv="LeaveOneOut(N=12 within each outer fold)",
        objective="maximize mean multi-output R2 across Flux, SMX, TRM, TET, and ERY",
        solver="scikit-learn MLPRegressor with solver='lbfgs'",
        regularization="L2 regularization through MLPRegressor alpha",
        standardization="z-score standardization in input and output space",
        scaling_source=str(args.scaling_source),
        selected_consensus_hyperparameters={
            activation: {
                "hidden_layer_sizes": list(consensus_configs[activation]["hidden_layer_sizes"]),
                "alpha": float(consensus_configs[activation]["alpha"]),
                "max_iter": int(consensus_configs[activation]["max_iter"]),
                "parameter_count": int(param_counts[activation]),
            }
            for activation in ACTIVATIONS
        },
        trainable_parameters_by_branch=param_counts,
        total_trainable_parameters=int(sum(param_counts.values())),
        scaling=stats,
    )

    write_summary_json(
        summary=summary,
        out_path=out_dir / "mesm_training_summary.json",
    )

    if args.predict:
        result = predict_single_from_exported_files(
            models_dir=models_dir,
            scaling_path=tables_dir / "mesm_scaling_stats.json",
            pressure=float(args.pressure),
            conc=float(args.conc),
            ph=float(args.ph),
        )

        print(json.dumps(result, indent=2))

    print("Saved GUI-compatible model files to:", models_dir)
    print("Saved reproducibility tables to:", tables_dir)
    print(metrics_oof.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())