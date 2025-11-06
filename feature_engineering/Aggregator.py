import pandas as pd
import numpy as np
from .extractors import *



def generate_features(battle_data: list[dict], flag_test: bool, difference: bool = True, tree: bool = True, divide_turns: bool = True) -> pd.DataFrame:
    """ Takes the raw battle data, generates all features, and joins them
    into a single DataFrame. 
    you can also select the right features for the specific model.
    
    """
    if tree :
        one_hot = False
    else:
        one_hot = True

    if flag_test:
        if tree: 
            df_list = [
            
                avg_team_vs_lead_stats(battle_data,difference=difference, test=flag_test),
                #avg_effectiveness2(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                #category_impact_score(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                #avg_stab_multiplier(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                faint_count_diff_extractor(battle_data, difference=difference,test=flag_test), # generalized by pokemon encoding
                avg_final_HP_pct(battle_data, difference=difference, test=flag_test),
                avg_boost_diff_per_turn(battle_data, test=flag_test),
                avg_stat_diff_per_turn(battle_data, test=flag_test, stats=['hp', 'atk', 'def', 'spa', 'spd', 'spe'], divide_turns=divide_turns),
                accuracy_avg(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                granular_turn_counts(battle_data, difference=difference, test=flag_test),
                ratio_category_diff(battle_data, difference=difference,test=flag_test),
                calculate_voluntary_swap_diff(battle_data, difference=difference, test=flag_test),
                team_hp_advantage_flip_count(battle_data, test=flag_test),
                damage_efficiency_ratio(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns), # maybe better without turns for trees
                pokemon_encoding(battle_data, one_hot=one_hot , test=flag_test),
                avg_approx_damage(battle_data, difference=difference ,test=flag_test)
            ]
        else:
            df_list = [
            
                avg_team_vs_lead_stats(battle_data,difference=difference, test=flag_test),
                avg_effectiveness2(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                #category_impact_score(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                #avg_stab_multiplier(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                faint_count_diff_extractor(battle_data, difference=difference,test=flag_test), # generalized by pokemon encoding
                avg_final_HP_pct(battle_data, difference=difference, test=flag_test),
                avg_boost_diff_per_turn(battle_data, test=flag_test),
                avg_stat_diff_per_turn(battle_data, test=flag_test, stats=['hp', 'atk', 'def', 'spa', 'spd', 'spe'], divide_turns=divide_turns),
                accuracy_avg(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                granular_turn_counts(battle_data, difference=difference, test=flag_test),
                ratio_category_diff(battle_data, difference=difference,test=flag_test),
                calculate_voluntary_swap_diff(battle_data, difference=difference, test=flag_test),
                team_hp_advantage_flip_count(battle_data, test=flag_test),
                damage_efficiency_ratio(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns), 
                pokemon_encoding(battle_data, one_hot=one_hot , test=flag_test),
                avg_approx_damage(battle_data, difference=difference , test=flag_test)
            ]           

    else:
        if tree:
            flag_test = True
            df_list = [
                avg_team_vs_lead_stats(battle_data,difference=difference, test=flag_test),
                #avg_effectiveness2(battle_data, difference=difference, divide_turns=divide_turns,test=flag_test),
                #category_impact_score(battle_data, difference=difference, test= flag_test, divide_turns=divide_turns),
                #avg_stab_multiplier(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                faint_count_diff_extractor(battle_data, difference=difference,test=flag_test),
                avg_final_HP_pct(battle_data, difference=difference, test=flag_test),
                avg_boost_diff_per_turn(battle_data, test=flag_test),
                avg_stat_diff_per_turn(battle_data, test=flag_test, stats=['hp', 'atk', 'def', 'spa', 'spd', 'spe'], divide_turns=divide_turns),
                accuracy_avg(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                granular_turn_counts(battle_data, difference=difference, test=flag_test),
                ratio_category_diff(battle_data, difference=difference ,test=flag_test),
                calculate_voluntary_swap_diff(battle_data, difference=difference,test=flag_test),
                team_hp_advantage_flip_count(battle_data, test=flag_test),
                damage_efficiency_ratio(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns), # maybe better without turns for trees
                pokemon_encoding(battle_data, one_hot=one_hot , test=flag_test),
                avg_approx_damage(battle_data, difference=difference , test=False)
            ]
        else:
            flag_test = True
            df_list = [
                avg_team_vs_lead_stats(battle_data,difference=difference, test=flag_test),
                avg_effectiveness2(battle_data, difference=difference, divide_turns=True,test=flag_test),
                #category_impact_score(battle_data, difference=difference, test= flag_test, divide_turns=divide_turns),
                #avg_stab_multiplier(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                faint_count_diff_extractor(battle_data, difference=difference,test=flag_test),
                avg_final_HP_pct(battle_data, difference=difference, test=flag_test),
                avg_boost_diff_per_turn(battle_data, test=flag_test),
                avg_stat_diff_per_turn(battle_data, test=flag_test, stats=['hp', 'atk', 'def', 'spa', 'spd', 'spe'], divide_turns=divide_turns),
                accuracy_avg(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns), 
                granular_turn_counts(battle_data, difference=difference, test=flag_test),
                ratio_category_diff(battle_data, difference=difference ,test=flag_test),
                calculate_voluntary_swap_diff(battle_data, difference=difference,test=flag_test),
                team_hp_advantage_flip_count(battle_data, test=flag_test),
                damage_efficiency_ratio(battle_data, difference=difference, test=flag_test, divide_turns=divide_turns),
                pokemon_encoding(battle_data, one_hot=one_hot , test=flag_test),
                avg_approx_damage(battle_data, difference=difference , test=False)
            ]            


    # Start with the first DataFrame in the list
    final_dataset = df_list[0]

    # Loop through the rest of the DataFrames (from the second one onwards)
    for df_to_merge in df_list[1:]:
        # Iteratively merge, using both keys to avoid duplicates
        final_dataset = pd.merge(final_dataset, df_to_merge, on=['battle_id'])

    return final_dataset