
#!/usr/bin/env python3
"""
Train a dialect classifier on wav2vec2.0-style embeddings and optionally run inference.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Set
import re

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score,
)

DEFAULT_LABEL_COL = "named_dialect"
META_NUMERIC_EXCLUDE: Set[str] = {"Unnamed: 0", "orig_index", "duration", "numeric_dialect"}

# Built-in aliases for multi-source training
ALIAS_TRAIN_TYPES: Dict[str, List[str]] = {
    "orig_aug": ["orig", "aug_pitch"],
    "orig_vc": ["orig", "vc_vc1"],
    "orig_multi_vc": ["orig", "vc_vc1", "vc_vc2", "vc_vc3", "vc_vc4"],
}

_NUMERIC_COL_RE = re.compile(r"^(\d+)(?:\.(\d+))?$")

def _scan_numeric_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (base, suffix) if name is numeric-like:
       "12"     -> (12, None)
       "12.1"   -> (12, 1)
       "12.10"  -> (12, 10)
    Otherwise -> (None, None)
    """
    m = _NUMERIC_COL_RE.fullmatch(str(name))
    if not m:
        return None, None
    base = int(m.group(1))
    suff = int(m.group(2)) if m.group(2) is not None else None
    return base, suff

def detect_mean_std_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [str(c) for c in df.columns]
    bases_to_names: Dict[int, Set[str]] = {}
    for c in cols:
        base, _ = _scan_numeric_name(c)
        if base is not None:
            bases_to_names.setdefault(base, set()).add(c)

    mean_cols: List[str] = []
    std_cols: List[str]  = []

    for base in sorted(bases_to_names.keys()):
        names = bases_to_names[base]
        # mean: exact "base"
        if str(base) in names:
            mean_cols.append(str(base))
        # std: exact "base.1" preferred, else the lowest suffix > 0 present
        if f"{base}.1" in names:
            std_cols.append(f"{base}.1")
        else:
            suffixes = []
            for n in names:
                b, s = _scan_numeric_name(n)
                if b == base and s is not None and s > 0:
                    suffixes.append(s)
            if suffixes:
                std_cols.append(f"{base}.{min(suffixes)}")

    return mean_cols, std_cols


def build_output_dir(base: str, split: str, data_type: str, out_dir: Optional[str] = None) -> str:
    if out_dir:
        out_path = out_dir
    else:
        out_path = os.path.join(base, split, data_type, "outputs")
    os.makedirs(out_path, exist_ok=True)
    return out_path

def _settings_fingerprint(args, feature_cols: List[str]) -> str:
    payload = json.dumps({
        "feature_mode": args.feature_mode,
        "use_pca": bool(args.use_pca),
        "pca_components": int(args.pca_components) if args.use_pca else 0,
        "C": float(args.C),
        "label_col": args.label_col,
        "n_features": len(feature_cols),
        "train_types": args.train_types,   # may be None or comma list
        "type": args.type,                 # top-level type/alias
        "split": args.split,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

def make_run_tag(args, feature_cols: List[str]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode = args.feature_mode
    pca = int(args.pca_components) if args.use_pca else 0
    nfe = len(feature_cols)
    sha = _settings_fingerprint(args, feature_cols)
    return f"{ts}_mode-{mode}_pca-{pca}_C-{args.C}_nfeats-{nfe}_sha-{sha}"

def _update_outputs_index(base_outputs_dir: str, run_dir_name: str) -> None:
    index_path = os.path.join(base_outputs_dir, "index.json")
    entry = {
        "run_dir": run_dir_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    data = {"runs": []}
    try:
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        if "runs" not in data or not isinstance(data["runs"], list):
            data = {"runs": []}
    except Exception:
        data = {"runs": []}
    data["runs"].append(entry)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_unique_run_dir(base_outputs_dir: str, run_tag: str) -> str:
    os.makedirs(base_outputs_dir, exist_ok=True)
    candidate = os.path.join(base_outputs_dir, run_tag)
    if not os.path.exists(candidate):
        os.makedirs(candidate, exist_ok=True)
        _update_outputs_index(base_outputs_dir, run_tag)
        return candidate
    i = 2
    while True:
        cand = os.path.join(base_outputs_dir, f"{run_tag}__{i}")
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=True)
            _update_outputs_index(base_outputs_dir, os.path.basename(cand))
            return cand
        i += 1

def point_latest_symlink(base_outputs_dir: str, run_dir: str) -> None:
    try:
        latest_link = os.path.join(base_outputs_dir, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(run_dir, latest_link)
    except Exception:
        pass

def _data_paths(base: str, split: str, t: str) -> Tuple[str, str]:
    folder = os.path.join(base, split, t)
    return (
        os.path.join(folder, "train_embeddings_metadata_layer1.csv"),
        os.path.join(folder, "dev_embeddings_metadata_layer1.csv"),
    )

def resolve_train_types(
    base: str, split: str, top_type: str, explicit_train_types: Optional[List[str]]
) -> List[str]:
    if explicit_train_types:
        train_types = explicit_train_types
    elif top_type in ALIAS_TRAIN_TYPES:
        train_types = ALIAS_TRAIN_TYPES[top_type]
    else:
        train_types = [top_type]
    missing = []
    for t in train_types:
        train_path, _ = _data_paths(base, split, t)
        if not os.path.exists(train_path):
            missing.append((t, train_path))
    if missing:
        lines = "\n".join([f" {t}: {p}" for t, p in missing])
        raise FileNotFoundError(f"Missing training file(s):\n{lines}")
    return train_types

def resolve_dev_type(
    base: str, split: str, top_type: str, dev_type_opt: Optional[str], default_train_type: str
) -> str:
    if dev_type_opt:
        dev_type = dev_type_opt
    else:
        _, dev_candidate = _data_paths(base, split, top_type)
        if os.path.exists(dev_candidate):
            dev_type = top_type
        else:
            dev_type = default_train_type
    _, dev_path = _data_paths(base, split, dev_type)
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Missing dev file for dev_type '{dev_type}': {dev_path}")
    return dev_type

def load_train_multiple(base: str, split: str, train_types: List[str]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for t in train_types:
        train_path, _ = _data_paths(base, split, t)
        df = pd.read_csv(train_path, sep="\t")
        df.columns = [str(c) for c in df.columns]
        df["_source_type"] = t
        dfs.append(df)
        print(f"[INFO] Loaded train from {t}: {len(df)} rows")
    big = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined train rows: {len(big)}")
    return big

def load_dev(base: str, split: str, dev_type: str) -> pd.DataFrame:
    _, dev_path = _data_paths(base, split, dev_type)
    df = pd.read_csv(dev_path, sep="\t")
    df.columns = [str(c) for c in df.columns]
    print(f"[INFO] Loaded dev from {dev_type}: {len(df)} rows")
    return df

def compute_feature_columns(dfs: List[pd.DataFrame], feature_mode: str = "mean_std") -> List[str]:
    def _sort_key(s: str):
        m = _NUMERIC_COL_RE.fullmatch(s)
        if m:
            base = int(m.group(1))
            suffix = int(m.group(2)) if m.group(2) is not None else -1
            return (0, base, suffix)
        return (1, s)

    per_df_sets: List[Set[str]] = []
    have_std_everywhere = True

    for df in dfs:
        mean_cols, std_cols = detect_mean_std_columns(df)
        if feature_mode == "mean":
            cols = set(mean_cols)
        elif feature_mode == "std":
            have_std_everywhere &= bool(std_cols)
            cols = set(std_cols)
        elif feature_mode == "mean_std":
            have_std_everywhere &= bool(std_cols)
            cols = set(mean_cols) | set(std_cols)
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        cols = cols - META_NUMERIC_EXCLUDE
        per_df_sets.append(cols)

    if feature_mode in ("std", "mean_std") and not have_std_everywhere:
        raise ValueError("Requested std features but at least one dataframe lacks '<base>.1' (or any suffixed std) columns.")

    candidates = set.intersection(*per_df_sets) if per_df_sets else set()
    if not candidates:
        raise ValueError("No common embedding columns found across provided dataframes.")

    return sorted(list(candidates), key=_sort_key)

def prepare_Xy(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].astype(np.float32).values
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    y = df[label_col].values
    return X, y

def get_id_cols(df: pd.DataFrame) -> Dict[str, pd.Series]:
    id_cols: Dict[str, pd.Series] = {}
    for key in ["DocID", "SpeakerID", "path", "Location"]:
        if key in df.columns:
            id_cols[key] = df[key]
    return id_cols

def train(args: argparse.Namespace) -> None:
    print(f"[INFO] Training with base={args.base} split={args.split} type={args.type}")
    explicit_train_types = [t.strip() for t in args.train_types.split(",")] if args.train_types else None
    train_types = resolve_train_types(args.base, args.split, args.type, explicit_train_types)
    dev_type = resolve_dev_type(args.base, args.split, args.type, args.dev_type, default_train_type=train_types[0])

    train_df = load_train_multiple(args.base, args.split, train_types)
    dev_df = load_dev(args.base, args.split, dev_type)

    feature_cols = compute_feature_columns([train_df, dev_df], feature_mode=args.feature_mode)
    print(f"[INFO] Using {len(feature_cols)} embedding features (mode={args.feature_mode}).")

    def inspect(name, X):
        print(f"[CHECK] {name}: shape={X.shape}")
        print(f" any NaN? {np.isnan(X).any()}")
        print(f" any Inf? {np.isinf(X).any()}")
        print(f" all finite? {np.isfinite(X).all()}")

    X_train, y_train = prepare_Xy(train_df, feature_cols, args.label_col)
    X_dev, y_dev = prepare_Xy(dev_df, feature_cols, args.label_col)

    def replace_nonfinite(X):
        X = np.array(X, copy=True)
        X[~np.isfinite(X)] = np.nan
        return X

    X_train = replace_nonfinite(X_train)
    X_dev = replace_nonfinite(X_dev)
    inspect("X_train raw", X_train)
    inspect("X_dev raw", X_dev)

    imp = SimpleImputer(strategy="constant", fill_value=0.0)
    X_train_imp = imp.fit_transform(X_train)
    X_dev_imp = imp.transform(X_dev)

    varth = VarianceThreshold(threshold=0.0)
    X_train_v = varth.fit_transform(X_train_imp)
    X_dev_v = varth.transform(X_dev_imp)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train_v)
    X_dev_s = scaler.transform(X_dev_v)

    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_dev_s = np.nan_to_num(X_dev_s, nan=0.0, posinf=0.0, neginf=0.0)

    steps: List[Tuple[str, object]] = []
    steps.append(("imputer", SimpleImputer(strategy="mean")))
    steps.append(("varth", VarianceThreshold(threshold=0.0)))
    steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    if args.use_pca:
        steps.append(("pca", PCA(n_components=args.pca_components, random_state=0)))
    steps.append(("clf", LinearSVC(class_weight="balanced", C=args.C, max_iter=args.max_iter)))
    pipeline = Pipeline(steps)

    print("[INFO] Fitting pipeline...", flush=True)
    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating on dev set...", flush=True)
    y_pred = pipeline.predict(X_dev)

    # PCA explained variance (if applicable)
    pca_metrics: Dict[str, object] = {}
    if args.use_pca and "pca" in pipeline.named_steps:
        pca: PCA = pipeline.named_steps["pca"]
        ratios = pca.explained_variance_ratio_
        cum = np.cumsum(ratios)
        total = float(np.sum(ratios))
        pca_metrics = {
            "pca_explained_variance_ratio": [float(x) for x in ratios],
            "pca_explained_variance_ratio_cumulative": [float(x) for x in cum],
            "pca_total_explained_variance": total,
            "pca_n_components": int(getattr(pca, "n_components_", getattr(pca, "n_components", len(ratios)))),
        }
        preview_n = min(10, len(ratios))
        print(f"[INFO] PCA components: {pca_metrics['pca_n_components']}")
        print(f"[INFO] PCA total explained variance: {total:.4f}")
        print("[INFO] PCA top components (ratio, cumulative):")
        for i in range(preview_n):
            print(f"  PC {i+1:3d}: {ratios[i]:.6f}  cum={cum[i]:.6f}")

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_dev, y_pred)),
        "macro_f1": float(f1_score(y_dev, y_pred, average="macro")),
        "classification_report": classification_report(y_dev, y_pred, digits=4),
        "n_train": int(X_train.shape[0]),
        "n_dev": int(X_dev.shape[0]),
        "n_features": int(X_train.shape[1]),
        "feature_mode": args.feature_mode,
        "use_pca": bool(args.use_pca),
        "pca_components": int(args.pca_components) if args.use_pca else None,
        "C": float(args.C),
        "label_col": args.label_col,
        "train_types": train_types,
        "dev_type": dev_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    metrics.update(pca_metrics)

    base_outputs_dir = build_output_dir(args.base, args.split, args.type, args.output_dir)
    run_tag = make_run_tag(args, feature_cols)
    out_dir = create_unique_run_dir(base_outputs_dir, run_tag)

    model_path = os.path.join(out_dir, "model.joblib")
    dump(pipeline, model_path)

    features_path = os.path.join(out_dir, "feature_columns.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols, "feature_mode": args.feature_mode}, f, ensure_ascii=False, indent=2)

    metrics_path = os.path.join(out_dir, "metrics_dev.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.use_pca and "pca" in pipeline.named_steps and "pca_explained_variance_ratio" in pca_metrics:
        pca_csv = os.path.join(out_dir, "pca_explained_variance.csv")
        df_pca = pd.DataFrame({
            "component_index": np.arange(1, len(pca_metrics["pca_explained_variance_ratio"]) + 1),
            "explained_variance_ratio": pca_metrics["pca_explained_variance_ratio"],
            "cumulative_ratio": pca_metrics["pca_explained_variance_ratio_cumulative"],
        })
        df_pca.to_csv(pca_csv, index=False)

    dev_ids = get_id_cols(dev_df)
    pred_df = pd.DataFrame({"pred_label": y_pred, "true_label": y_dev})
    for k, s in dev_ids.items():
        pred_df[k] = s
    pred_out = os.path.join(out_dir, "dev_predictions.csv")
    pred_df.to_csv(pred_out, index=False)

    point_latest_symlink(base_outputs_dir, out_dir)

    print("[INFO] Saved:")
    print(f" Run dir: {out_dir}")
    print(f" Model: {model_path}")
    print(f" Features JSON: {features_path}")
    print(f" Dev metrics: {metrics_path}")
    if args.use_pca and "pca" in pipeline.named_steps:
        print(f" PCA CSV saved")
    print(f" Dev predictions: {pred_out}")

def infer(args: argparse.Namespace) -> None:
    print(f"[INFO] Inference with model={args.model} input={args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.features):
        raise FileNotFoundError(f"Feature column JSON not found: {args.features}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    pipeline = load(args.model)
    with open(args.features, "r", encoding="utf-8") as f:
        j = json.load(f)
        feature_cols = j["feature_columns"]
        saved_mode = j.get("feature_mode")
        if saved_mode and args.feature_mode and saved_mode != args.feature_mode:
            print(f"[WARN] Feature mode mismatch: saved={saved_mode} runtime={args.feature_mode}. Proceeding with saved features.", flush=True)

    df = pd.read_csv(args.input, sep="\t")
    df.columns = [str(c) for c in df.columns]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input file missing expected feature columns: {missing[:10]} ... (total {len(missing)})")

    X = df[feature_cols].astype(np.float32).values
    y_true = df[args.label_col].values if args.label_col and args.label_col in df.columns else None

    print("[INFO] Predicting...", flush=True)
    y_pred = pipeline.predict(X)

    out_rows: Dict[str, object] = {"pred_label": y_pred}
    if y_true is not None:
        out_rows["true_label"] = y_true
    ids = get_id_cols(df)
    for k, s in ids.items():
        out_rows[k] = s
    pred_df = pd.DataFrame(out_rows)
    pred_out = args.pred_out if args.pred_out else os.path.join(os.path.dirname(args.input), "predictions.csv")
    pred_df.to_csv(pred_out, index=False)
    print(f"[INFO] Saved predictions: {pred_out}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/infer dialect classifier on wav2vec embeddings")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Training arguments
    parser.add_argument("--base", type=str, help="Base folder (e.g., /scratch/user/speaker_partitions)")
    parser.add_argument("--split", type=str, help="Split name (e.g., split1)")
    parser.add_argument("--type", type=str, help="Top-level type or alias (e.g., orig, orig_aug, orig_vc, orig_multi_vc)")
    parser.add_argument("--train-types", type=str, default=None,
                        help="Comma-separated list of type folders to use for training (e.g., 'orig,aug_pitch'). Overrides aliases.")
    parser.add_argument("--dev-type", type=str, default=None,
                        help="Type folder to use for dev (default: '--type' if present, else first training type).")
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL, help="Target label column")
    parser.add_argument("--use-pca", action="store_true", help="Enable PCA before classifier")
    parser.add_argument("--pca-components", type=int, default=128, help="Number of PCA components if enabled")
    parser.add_argument("--C", type=float, default=1.0, help="LinearSVC regularization strength")
    parser.add_argument("--max-iter", type=int, default=5000, help="LinearSVC max iterations")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional custom output directory")
    parser.add_argument("--feature-mode", type=str, choices=["mean", "std", "mean_std"], default="mean_std",
                        help="Which embedding features to use")

    # Inference subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference with a saved model")
    infer_parser.add_argument("--model", type=str, required=True, help="Path to saved model (.joblib)")
    infer_parser.add_argument("--features", type=str, required=True, help="Path to feature_columns.json")
    infer_parser.add_argument("--input", type=str, required=True, help="Path to input CSV/TSV (e.g., test file)")
    infer_parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL,
                              help="Label column (optional; if present, included in output)")
    infer_parser.add_argument("--pred-out", type=str, default=None, help="Output predictions CSV path")
    infer_parser.add_argument("--feature-mode", type=str, choices=["mean", "std", "mean_std"], default="mean_std",
                              help="Feature mode used during training (for information only; JSON wins)")

    return parser

def main():
    parser = build_arg_parser()
    # Loop over multiple types (optional)
    parser.add_argument("--types", type=str, default=None,
                        help="Comma-separated list of top-level types/aliases to train sequentially (overrides --type).")
    args = parser.parse_args()

    if args.command == "infer":
        infer(args)
        return

    if args.types:
        type_list = [t.strip() for t in args.types.split(",") if t.strip()]
        if not type_list:
            print("[ERROR] --types provided but empty.", file=sys.stderr)
            sys.exit(2)
        if any(x is None for x in [args.base, args.split]):
            print("[ERROR] Missing --base/--split for --types.", file=sys.stderr)
            sys.exit(2)
        print(f"[INFO] Looping over types: {type_list}")
        for t in type_list:
            per_args = argparse.Namespace(**vars(args))
            per_args.type = t
            print(f"\n[INFO] === Training for type='{t}' ===")
            train(per_args)
        print("\n[INFO] All types finished.")
        return

    required = [args.base, args.split, args.type]
    if any(x is None for x in required):
        print("[ERROR] Missing required training args: --base --split --type", file=sys.stderr)
        sys.exit(2)
    train(args)

if __name__ == "__main__":
    main()

