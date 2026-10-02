import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import (
    MODELS_DIR,
    NSL_KDD_TEST_PATH,
    NSL_KDD_TRAIN_PATH,
    PROCESSED_DATA_DIR,
    UNSW_NB15_TEST_PATH,
    UNSW_NB15_TRAIN_PATH,
    logger,
)

# raw files have no header row
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
NSL_KDD_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# attack families from Dhanabal & Shantharajah (2015)
NSL_KDD_ATTACK_CATEGORY = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "apache2": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

UNSW_NB15_CATEGORICAL_COLS = ["proto", "service", "state"]


def load_nsl_kdd_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading NSL-KDD train set from {NSL_KDD_TRAIN_PATH}")
    train_df = pd.read_csv(NSL_KDD_TRAIN_PATH, names=NSL_KDD_COLUMNS)
    logger.info(f"Loading NSL-KDD test set from {NSL_KDD_TEST_PATH}")
    test_df = pd.read_csv(NSL_KDD_TEST_PATH, names=NSL_KDD_COLUMNS)

    # difficulty is metadata, not a traffic feature
    train_df = train_df.drop(columns=["difficulty"])
    test_df = test_df.drop(columns=["difficulty"])

    for df in (train_df, test_df):
        df["attack_category"] = df["label"].map(NSL_KDD_ATTACK_CATEGORY)
        df["binary_label"] = (df["label"] != "normal").astype(int)

    logger.debug(f"NSL-KDD train shape={train_df.shape}, test shape={test_df.shape}")
    return train_df, test_df


def load_unsw_nb15_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading UNSW-NB15 train set from {UNSW_NB15_TRAIN_PATH}")
    train_df = pd.read_csv(UNSW_NB15_TRAIN_PATH)
    logger.info(f"Loading UNSW-NB15 test set from {UNSW_NB15_TEST_PATH}")
    test_df = pd.read_csv(UNSW_NB15_TEST_PATH)

    train_df = train_df.drop(columns=["id"])
    test_df = test_df.drop(columns=["id"])

    for df in (train_df, test_df):
        df["attack_category"] = df["attack_cat"].fillna("Normal").str.strip()
        df["binary_label"] = df["label"].astype(int)

    logger.debug(f"UNSW-NB15 train shape={train_df.shape}, test shape={test_df.shape}")
    return train_df, test_df


def inspect_dataset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, categorical_cols: list[str]
) -> dict:
    duplicate_rows = int(train_df.duplicated().sum())
    missing_values = int(train_df.isnull().sum().sum() + test_df.isnull().sum().sum())

    info = {
        "name": dataset_name,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "num_features": train_df.shape[1] - 3,
        "categorical_features": categorical_cols,
        "numeric_features": [
            c for c in train_df.columns
            if c not in categorical_cols and c not in ("label", "attack_cat", "binary_label", "attack_category")
        ],
        "duplicate_rows_in_train": duplicate_rows,
        "missing_values_found": missing_values,
        "class_distribution_train": {
            "normal": int((train_df["binary_label"] == 0).sum()),
            "attack": int((train_df["binary_label"] == 1).sum()),
        },
        "class_distribution_test": {
            "normal": int((test_df["binary_label"] == 0).sum()),
            "attack": int((test_df["binary_label"] == 1).sum()),
        },
        "attack_categories_train": train_df["attack_category"].value_counts().to_dict(),
    }

    logger.info(
        f"[{dataset_name}] train={info['train_records']} test={info['test_records']} "
        f"features={info['num_features']} duplicates_in_train={duplicate_rows} "
        f"missing_values={missing_values}"
    )
    logger.info(f"[{dataset_name}] train class balance: {info['class_distribution_train']}")

    if missing_values:
        logger.warning(f"[{dataset_name}] found {missing_values} missing values")

    return info


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str],
    dataset_name: str,
) -> dict:
    """Fit one-hot encoding + scaling on the train set and transform both splits."""
    before = len(train_df)
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    removed = before - len(train_df)
    if removed:
        logger.info(f"[{dataset_name}] removed {removed} duplicate rows from train set")

    numeric_cols = [
        c for c in train_df.columns
        if c not in categorical_cols and c not in ("label", "attack_cat", "binary_label", "attack_category")
    ]

    X_train_raw = train_df[categorical_cols + numeric_cols]
    X_test_raw = test_df[categorical_cols + numeric_cols]
    y_train = train_df["binary_label"].to_numpy()
    y_test = test_df["binary_label"].to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    logger.info(f"[{dataset_name}] fitting preprocessor (one-hot + scaling) on training data")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_normal = X_train[y_train == 0]
    logger.info(
        f"[{dataset_name}] preprocessed X_train={X_train.shape} X_test={X_test.shape} "
        f"X_train_normal={X_train_normal.shape} (used to train the VAE)"
    )

    test_raw_preview = test_df[categorical_cols + ["attack_category"]].reset_index(drop=True)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_normal": X_train_normal,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "test_raw_preview": test_raw_preview,
    }


def save_processed(dataset_name: str, processed: dict, dataset_info: dict) -> None:
    out_dir = PROCESSED_DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "train.npz", X=processed["X_train"], y=processed["y_train"])
    np.savez_compressed(out_dir / "test.npz", X=processed["X_test"], y=processed["y_test"])
    np.savez_compressed(out_dir / "train_normal.npz", X=processed["X_train_normal"])
    processed["test_raw_preview"].to_csv(out_dir / "test_raw_preview.csv", index=False)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(processed["feature_names"], f, indent=2)

    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    preprocessor_path = MODELS_DIR / f"{dataset_name}_preprocessor.joblib"
    joblib.dump(processed["preprocessor"], preprocessor_path)

    logger.info(f"[{dataset_name}] saved processed arrays and preprocessor to {out_dir} / {preprocessor_path}")


def load_processed(dataset_name: str) -> dict:
    out_dir = PROCESSED_DATA_DIR / dataset_name

    train = np.load(out_dir / "train.npz")
    test = np.load(out_dir / "test.npz")
    train_normal = np.load(out_dir / "train_normal.npz")
    test_raw_preview = pd.read_csv(out_dir / "test_raw_preview.csv")

    with open(out_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    preprocessor = joblib.load(MODELS_DIR / f"{dataset_name}_preprocessor.joblib")

    return {
        "X_train": train["X"],
        "y_train": train["y"],
        "X_test": test["X"],
        "y_test": test["y"],
        "X_train_normal": train_normal["X"],
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "test_raw_preview": test_raw_preview,
    }


def run(dataset_name: str) -> None:
    if dataset_name == "nsl_kdd":
        train_df, test_df = load_nsl_kdd_raw()
        categorical_cols = NSL_KDD_CATEGORICAL_COLS
    elif dataset_name == "unsw_nb15":
        train_df, test_df = load_unsw_nb15_raw()
        categorical_cols = UNSW_NB15_CATEGORICAL_COLS
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = inspect_dataset(train_df, test_df, dataset_name, categorical_cols)
    processed = preprocess(train_df, test_df, categorical_cols, dataset_name)
    save_processed(dataset_name, processed, dataset_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NSL-KDD and/or UNSW-NB15")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "all"], default="all")
    args = parser.parse_args()

    datasets = ["nsl_kdd", "unsw_nb15"] if args.dataset == "all" else [args.dataset]
    for name in datasets:
        run(name)
