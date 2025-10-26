import pandas as pd
import numpy as np
from .extractors import *



def generate_features(battle_data: list[dict], flag_test: bool) -> pd.DataFrame:
    """ Takes the raw battle data, generates all features, and joins them
    into a single DataFrame. """
    
    if flag_test:

        df_list = [
            avg_effectiveness_1_1(battle_data, difference=True, include_status_moves=False, test=flag_test),
            category_impact_score(battle_data, difference=True, test=flag_test),
            avg_stab_multiplier(battle_data, difference=True, test=flag_test),
            avg_final_HP_pct(battle_data, difference=True, test=flag_test),
            avg_boost_diff_per_turn(battle_data, difference=True, test=flag_test),
            avg_stat_diff_per_turn(battle_data, test=flag_test, stats=['hp', 'atk', 'def', 'spa', 'spd', 'spe']),
            accuracy_basepower_avg(battle_data, difference=True, test=flag_test)
        ]

    else:
        flag_test = True
        df_list = [
            avg_effectiveness_1_1(battle_data, difference=True, include_status_moves=False, test=flag_test),
            category_impact_score(battle_data, difference=True, test= flag_test),
            avg_stab_multiplier(battle_data, difference=True, test=flag_test),
            avg_final_HP_pct(battle_data, difference=True, test=flag_test),
            avg_boost_diff_per_turn(battle_data, difference=True, test=flag_test),
            avg_stat_diff_per_turn(battle_data, test=flag_test, stats=['hp', 'atk', 'def', 'spa', 'spd', 'spe']),
            accuracy_basepower_avg(battle_data, difference=True, test=False)
        ]


    if not df_list:
        print("Warning: No feature DataFrames were generated.")
        return pd.DataFrame()

    # Start with the first DataFrame in the list
    final_dataset = df_list[0]

    # Loop through the rest of the DataFrames (from the second one onwards)
    for df_to_merge in df_list[1:]:
        # Iteratively merge, using both keys to avoid duplicates
        final_dataset = pd.merge(final_dataset, df_to_merge, on=['battle_id'])

    return final_dataset