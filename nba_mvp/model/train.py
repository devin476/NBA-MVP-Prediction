import os
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from nba_mvp.utils.features import engineer_features


def load_data(data_dir='nba_mvp/data'):
    csv_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and '2024_25' not in f
    ])
    df_all = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in csv_files], ignore_index=True)
    return df_all


def prepare_data(df_all):
    df = engineer_features(df_all)
    y = df["MVP"].astype(int)
    drop_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "SEASON", "MVP"]
    low_impact_cols = [
        "FTA", "REB", "FG3A", "FG3M", "BLKA", "FGA", "PF", "W_PCT",
        "DREB_RANK", "FTA_RANK", "STL", "FTM", "PTS", "BLKA_RANK"
    ]

    X = df.drop(columns=drop_cols + low_impact_cols, errors="ignore")
    X = X.drop(columns=[col for col in X.columns if "FANTASY" in col.upper()], errors="ignore")
    X = X.select_dtypes(include="number")


    return X, y


def train_model(data_dir='nba_mvp/data'):
    df = load_data(data_dir)
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict probabilities and determine threshold
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    balanced_idx = np.argmin(np.abs(precision[:-1] - recall[:-1]))
    balanced_thresh = thresholds[balanced_idx]
    y_pred_balanced = (probs > balanced_thresh).astype(int)

    # Save threshold
    with open(os.path.join('nba_mvp', 'model', 'threshold.json'), 'w') as f:
        json.dump({"threshold": float(balanced_thresh)}, f)

    print(f"\nBalanced threshold: {balanced_thresh:.4f}")
    print("- Classification Report at balanced precision ≈ recall -")
    print(classification_report(y_test, y_pred_balanced, digits=4))

    # Save model and features
    model_path = os.path.join('nba_mvp', 'model', 'mvp_model.joblib')
    feature_path = os.path.join('nba_mvp', 'model', 'features.json')
    joblib.dump(model, model_path)
    with open(feature_path, 'w') as f:
        json.dump(list(X.columns), f)

    print(f"\nModel saved to: {model_path}")
    print(f"Feature list saved to: {feature_path}")


if __name__ == "__main__":
    train_model()
