import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_mvp.data.mvps_by_season import mvps_by_season

def get_season_stats(season, min_games=15):
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame'
    )
    df = stats.get_data_frames()[0]
    df = df[df['GP'] >= min_games].copy()
    df['SEASON'] = season
    return df

def update_season_data(start_year=1996, end_year=2025, data_dir='nba_mvp/data'):
    os.makedirs(data_dir, exist_ok=True)

    for year in range(start_year, end_year):
        season = f"{year}-{str(year + 1)[-2:]}"
        try:
            print(f"Processing season: {season}")
            df = get_season_stats(season)

            mvp_id = mvps_by_season.get(season)
            df['MVP'] = df['PLAYER_ID'].apply(lambda pid: 1 if pid == mvp_id else 0)

            filename = os.path.join(data_dir, f"nba_player_stats_{season.replace('-', '_')}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved to {filename}")
            time.sleep(1)

        except Exception as e:
            print(f"Error with {season}: {e}")

    print("Data update complete.")
