import os
import json
import argparse
import joblib
import pandas as pd
from nba_mvp.utils.features import engineer_features


def run_prediction(season: str = "2024-25",
                   data_dir: str = "nba_mvp/data",
                   csv_path: str | None = None) -> None:
    """Run MVP predictions for a given season or CSV file."""
    model_path = os.path.join("nba_mvp", "model", "mvp_model.joblib")
    feature_path = os.path.join("nba_mvp", "model", "features.json")
    threshold_path = os.path.join("nba_mvp", "model", "threshold.json")

    if csv_path is None:
        csv_file = f"nba_player_stats_{season.replace('-', '_')}.csv"
        csv_path = os.path.join(data_dir, csv_file)

    # Load model and features
    model = joblib.load(model_path)
    with open(feature_path, "r") as f:
        feature_list = json.load(f)

    # Load threshold (fallback to 0.5)
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)

    # Load and prepare data
    df = pd.read_csv(csv_path)
    df = engineer_features(df)

    drop_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "SEASON", "MVP"]
    low_impact_cols = [
        "FTA", "REB", "FG3A", "FG3M", "BLKA", "FGA", "PF", "W_PCT",
        "DREB_RANK", "FTA_RANK", "STL", "FTM", "PTS", "BLKA_RANK"
    ]
    df_features = df.drop(columns=drop_cols + low_impact_cols, errors="ignore")
    df_features = df_features.drop(columns=[col for col in df_features.columns if "FANTASY" in col.upper()], errors="ignore")
    df_features = df_features.select_dtypes(include="number")
    df_features = df_features.reindex(columns=feature_list, fill_value=0)

    # Predict
    probs = model.predict_proba(df_features)[:, 1]
    preds = (probs > threshold).astype(int)

    # Format output
    output = df[["PLAYER_NAME", "PLAYER_ID", "TEAM_ABBREVIATION", "SEASON"]].copy()
    output["MVP_Prob"] = probs
    output["Predicted_MVP"] = preds

    print("\nTop 10 MVP Candidates:")
    print(output.sort_values(by="MVP_Prob", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict NBA MVP probabilities")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a CSV file of player stats. Overrides --season and --data-dir",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        help="Season to predict if --csv is not provided",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default="nba_mvp/data",
        help="Directory containing season CSVs",
    )

    args = parser.parse_args()
    run_prediction(season=args.season, data_dir=args.data_dir, csv_path=args.csv)
