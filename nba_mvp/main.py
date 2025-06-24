import argparse
from nba_mvp.model.train import train_model
from nba_mvp.model.predict import run_prediction
from nba_mvp.data.update_data import update_season_data

def main():
    parser = argparse.ArgumentParser(description="NBA MVP Prediction CLI")

    parser.add_argument(
        "--mode", 
        choices=["train", "predict", "update-data"], 
        required=True,
        help="Mode to run: train the model, predict MVPs, or update data"
    )
    parser.add_argument(
        "--season", 
        type=str, 
        default="2024-25",
        help="Season to predict (used only in predict mode)"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data",
        help="Path to the data directory"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(data_dir=args.data)
    elif args.mode == "predict":
        run_prediction(season=args.season, data_dir=args.data)
    elif args.mode == "update-data":
        update_season_data(data_dir=args.data)

if __name__ == "__main__":
    main()
