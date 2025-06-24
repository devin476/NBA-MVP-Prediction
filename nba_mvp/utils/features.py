import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #Engineered features
    df['FGM_PER_MIN'] = df['FGM'] / df['MIN'].replace(0, 1)
    df['USAGE_PROXY'] = df['AST'] + df['TOV']
    df['PTS_PER_WIN'] = df['PTS'] / (df['W'] + 1e-5)
    df['DOMINANCE_SCORE'] = (
        df['PTS'] +
        1.2 * df['REB'] +
        1.5 * df['AST'] +
        3.0 * df['STL'] +
        3.0 * df['BLK'] -
        2.0 * df['TOV']
    )

    return df

def get_feature_list(df):
    """Extracts and returns the feature list from a given dataframe."""
    drop_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON', 'MVP']
    low_impact_cols = ['FTA', 'REB', 'FG3A', 'FG3M', 'BLKA', 'FGA', 'PF', 'W_PCT', 'DREB_RANK', 'FTA_RANK', 'STL', 'FTM', 'PTS', 'BLKA_RANK']
    
    df = df.drop(columns=drop_cols + low_impact_cols, errors='ignore')
    df = df.drop(columns=[col for col in df.columns if 'FANTASY' in col.upper()], errors='ignore')
    df = df.select_dtypes(include='number')
    
    return df.columns.to_list()
