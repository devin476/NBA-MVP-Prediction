# NBA-MVP-Prediction
XGBoost model with 96% Top-1 Accuracy

This project builds a machine learning model to predict the NBA MVP (Most Valuable Player) based on player statistics from the 1996–2024 NBA seasons. It includes automated feature engineering, model training with XGBoost for exploring MVP predictions.

## Project Structure

nba_mvp/  
├── __init__.py  
├── data/                      #Historical NBA player data from 1996-2024 (CSV format)  
│  
├── model/  
│   ├── __init__.py  
│   ├── train.py               #Trains and saves the MVP prediction model  
│   ├── predict.py             #Uses trained model to predict MVP probabilities  
│   ├── mvp_model.joblib       #Saved XGBoost model  
│   └── features.json          #List of features used in the model  
│  
├── utils/  
│   ├── __init__.py  
│   └── features.py            #Feature engineering functions  
│   
│  
├── main.py                    #CLI to run training or prediction  
├── requirements.txt           #Python dependencies  
└── README.md                  #Documentation

## Installation

1. Clone the repository:

git clone https://github.com/devin476/NBA-MVP-Prediction  
cd NBA-MVP-Prediction

2. (Optional) Create a virtual environment:

python -m venv venv  
source venv/bin/activate  (on Windows: venv\Scripts\activate)

3. Install dependencies:

pip install -r requirements.txt

## Usage

## Download the NBA Season data

Run:

python -m nba_mvp.main --mode update-data --data nba_mvp/data


### Train the MVP Prediction Model

Ensure CSVs are stored in nba_mvp/data/ and named like nba_player_stats_YYYY_YY.csv.

Run:

python -m nba_mvp.model.train

This will save:
- The model to nba_mvp/model/mvp_model.joblib  
- The feature list to nba_mvp/model/features.json
- The threshold to nba_mvp/model/threshold.json

### Predict MVP Probabilities

Run predictions on new season data:

python -m nba_mvp.model.predict --csv nba_mvp/data/nba_player_stats_2024_25.csv

It will output a list of players with their predicted MVP probabilities.


## Features

- Uses over 25 years of NBA data  
- Feature engineering includes:
  - Dominance Score
  - Field Goals Made Per Min
  - Points Per Win
  - Year over Year Points Delta 
- XGBoost classifier with class imbalance handling  
- Threshold optimization using precision-recall balance   

## Requirements

Major libraries:

- pandas  
- numpy  
- xgboost  
- scikit-learn  
- joblib 
- matplotlib

Install with:

pip install -r requirements.txt

## Contact

Feel free to reach out via GitHub for suggestions or collaboration.