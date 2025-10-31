# feature_engineering/extractors.py

import numpy as np
import pandas as pd
from .utils import get_dict_def_types, get_dict_attacker_types, get_dict_base_stats, effectiveness, get_dict_base_stats1

def avg_effectiveness_1(data: list[dict], difference=False, test=False) -> pd.DataFrame:
    """ Given the database 
        calculate the average effective multiplier of all moves used by P1 and P2 throughout the turns.
    """
    pokemon_def_types = get_dict_def_types(data)
    final=[]
    for battle in data:
        total_turns=0 
        total_effectiveness_p1 = 0.0
        total_effectiveness_p2 = 0.0
        for turn in battle['battle_timeline']:
            p1_pokemon_state= turn.get('p1_pokemon_state', {}) # take the p1_pokemon_state dict
            p2_pokemon_state= turn.get('p2_pokemon_state', {})

            name1 = p1_pokemon_state.get('name') # take the name of the pokemon insidide p1_pokemon_state dict
            name2 = p2_pokemon_state.get('name')

            type_def_1 = pokemon_def_types.get(name1)
            type_def_2 = pokemon_def_types.get(name2)

            p1_move_details = turn.get('p1_move_details', {}) # no move is considered as +0: penalize the final avg
            p2_move_details = turn.get('p2_move_details', {})
            
            if p1_move_details:
                type_att_1 = p1_move_details.get('type')
                total_effectiveness_p1 += effectiveness(type_att_1, type_def_2)

            if p2_move_details:
                type_att_2 = p2_move_details.get('type')
                total_effectiveness_p2 += effectiveness(type_att_2, type_def_1)

            total_turns += 1
        
        avg_eff_p1 = total_effectiveness_p1 / total_turns 
        avg_eff_p2 = total_effectiveness_p2 / total_turns

        result = {'battle_id': battle['battle_id']}
        
        if difference:
            result['avg_effectiveness_diff'] = avg_eff_p1 - avg_eff_p2
        else: 
            result['avg_effectiveness_p1'] = avg_eff_p1
            result['avg_effectiveness_p2'] = avg_eff_p2
        
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
    return pd.DataFrame(final)


def avg_effectiveness_1_1(data: list[dict], difference=False, include_status_moves=True, test=False) -> pd.DataFrame:
    """ Given the database
        calculate the average effective multiplier of all moves used by P1 and P2 throughout the turns.
        This version includes an option to include or exclude Status moves from the calculation, given the fact that they do not deal damage.
        
        Args:
            data (list): The list of battle dictionaries.
            difference (bool): If True, returns the difference (P1 - P2) in scores.
            include_status_moves (bool): If True, Status moves are included in the average (default behavior).
                                         If False, only Physical and Special moves contribute to total_effectiveness.
    """
    pokemon_def_types = get_dict_def_types(data)
    final=[]
    for battle in data:
        total_turns=0 
        total_effectiveness_p1 = 0.0
        total_effectiveness_p2 = 0.0
        for turn in battle['battle_timeline']:
            p1_pokemon_state= turn.get('p1_pokemon_state', {})
            p2_pokemon_state= turn.get('p2_pokemon_state', {})

            name1 = p1_pokemon_state.get('name')
            name2 = p2_pokemon_state.get('name')

            type_def_1 = pokemon_def_types.get(name1)
            type_def_2 = pokemon_def_types.get(name2)

            p1_move_details = turn.get('p1_move_details', {})
            p2_move_details = turn.get('p2_move_details', {})
            
            # --- P1 Calculation ---
            if p1_move_details:
                category_1 = p1_move_details.get('category').upper()
                
                # Check the new flag: only proceed if it's a damaging move OR status moves are included
                if include_status_moves or category_1 in ('PHYSICAL', 'SPECIAL'):
                    type_att_1 = p1_move_details.get('type')
                    total_effectiveness_p1 += effectiveness(type_att_1, type_def_2)

            # --- P2 Calculation ---
            if p2_move_details:
                category_2 = p2_move_details.get('category').upper()
                
                # Check the new flag: only proceed if it's a damaging move OR status moves are included
                if include_status_moves or category_2 in ('PHYSICAL', 'SPECIAL'):
                    type_att_2 = p2_move_details.get('type')
                    total_effectiveness_p2 += effectiveness(type_att_2, type_def_1)

            total_turns += 1
        
        avg_eff_p1 = total_effectiveness_p1 / total_turns 
        avg_eff_p2 = total_effectiveness_p2 / total_turns

        result = {'battle_id': battle['battle_id']}
            
        if difference:
            result['avg_effectiveness_diff'] = avg_eff_p1 - avg_eff_p2
        else: 
            result['avg_effectiveness_p1'] = avg_eff_p1
            result['avg_effectiveness_p2'] = avg_eff_p2
        
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
    return pd.DataFrame(final)

# Extended version of avg_effectiveness with turn segmentation
def avg_effectiveness2(data: list[dict], difference: bool = False, divide_turns: bool = False, test: bool = False) -> pd.DataFrame:
    """ Given the database, dict with pokemon name:list of types 
    calculate the average effectiveness of all moves used by P1 and P2 in each battle.
    
    :param data: List of battle dictionaries.
    :param difference: If True, returns the difference (P1 - P2).
    :param divide_turns: If True, computes separate averages for the first 10, middle 10, and last 10 turns.
    :return: A pandas DataFrame with the calculated average effectiveness data.
    """
    pokemon_def_types= get_dict_def_types(data)

    final = []
    
    # Define turn segmentation boundaries
    TURN_SEGMENTS = {
        'first_10': (0, 10),
        'middle_10': (10, 20),
        'last_10': (-10, None) # Use slicing logic: last 10 turns
    }

    for battle in data:
        timeline = battle['battle_timeline']
        total_turns = len(timeline)
        
        # --- Logic for Non-Divided Turns (Original Functionality) ---
        if not divide_turns:
            total_effectiveness_p1 = 0.0
            total_effectiveness_p2 = 0.0
            
            for turn in timeline:
                p1_pokemon_state = turn.get('p1_pokemon_state', {})
                p2_pokemon_state = turn.get('p2_pokemon_state', {})

                name1 = p1_pokemon_state.get('name')
                name2 = p2_pokemon_state.get('name')

                type_def_1 = pokemon_def_types.get(name1)
                type_def_2 = pokemon_def_types.get(name2)

                p1_move_details = turn.get('p1_move_details', {})
                p2_move_details = turn.get('p2_move_details', {})
                
                if p1_move_details:
                    type_att_1 = p1_move_details.get('type')
                    # NOTE: Assuming effectiveness(type_att, type_def) is defined elsewhere
                    total_effectiveness_p1 += effectiveness(type_att_1, type_def_2)

                if p2_move_details:
                    type_att_2 = p2_move_details.get('type')
                    total_effectiveness_p2 += effectiveness(type_att_2, type_def_1)
            
            if total_turns == 0:
                 avg_eff_p1, avg_eff_p2 = 0.0, 0.0
            else:
                avg_eff_p1 = total_effectiveness_p1 / total_turns 
                avg_eff_p2 = total_effectiveness_p2 / total_turns
            
            battle_result = {'battle_id': battle['battle_id']}
            if difference:
                battle_result['avg_diff'] = avg_eff_p1 - avg_eff_p2
            else: 
                battle_result['avg_effectiveness_p1'] = avg_eff_p1
                battle_result['avg_effectiveness_p2'] = avg_eff_p2
            
            if not test:
                battle_result['player_won'] = battle['player_won']
            
            final.append(battle_result)

        # --- Logic for Divided Turns (`divide_turns=True`) ---
        else:
            battle_data = {'battle_id': battle['battle_id']}
            
            for segment_name, (start, end) in TURN_SEGMENTS.items():
                # Determine the slice of the timeline for the current segment
                # Handles the different indexing conventions 
                if start is not None and end is None and start < 0:
                    # Last 10 turns
                    segment_timeline = timeline[start:]
                elif start is not None and end is not None:
                    # First 10 or Middle 10
                    segment_timeline = timeline[start:end]
                else:
                    segment_timeline = [] # Should not happen with defined segments

                segment_turns = len(segment_timeline)
                segment_eff_p1 = 0.0
                segment_eff_p2 = 0.0
                
                # Calculate effectiveness for the segment
                for turn in segment_timeline:
                    p1_pokemon_state = turn.get('p1_pokemon_state', {})
                    p2_pokemon_state = turn.get('p2_pokemon_state', {})

                    name1 = p1_pokemon_state.get('name')
                    name2 = p2_pokemon_state.get('name')

                    type_def_1 = pokemon_def_types.get(name1)
                    type_def_2 = pokemon_def_types.get(name2)

                    p1_move_details = turn.get('p1_move_details', {})
                    p2_move_details = turn.get('p2_move_details', {})
                    
                    if p1_move_details:
                        type_att_1 = p1_move_details.get('type')
                        segment_eff_p1 += effectiveness(type_att_1, type_def_2)

                    if p2_move_details:
                        type_att_2 = p2_move_details.get('type')
                        segment_eff_p2 += effectiveness(type_att_2, type_def_1)
                
                # Calculate average effectiveness for the segment
                if segment_turns == 0:
                    avg_eff_p1 = 0.0
                    avg_eff_p2 = 0.0
                else:
                    avg_eff_p1 = segment_eff_p1 / segment_turns
                    avg_eff_p2 = segment_eff_p2 / segment_turns
                
                # Store results for the segment
                if difference:
                    battle_data[f'avg_diff_{segment_name}'] = avg_eff_p1 - avg_eff_p2
                else:
                    battle_data[f'avg_effectiveness_p1_{segment_name}'] = avg_eff_p1
                    battle_data[f'avg_effectiveness_p2_{segment_name}'] = avg_eff_p2
            
            if not test:
                battle_data['player_won'] = battle['player_won']
            
            final.append(battle_data)

    return pd.DataFrame(final)


def category_impact_score(data: list[dict], difference=False, test=False):
    """Given the database, calculate the average category impact score of all moves used by P1 and P2 throughout the turns.
       Category impact score is defined as: base_atk/base_def for Physical moves and base_spa/base_spd for Special moves.
    """

    dict_base_stats = get_dict_base_stats(data)
    final = []
    for battle in data:
        p1_score = 0
        p2_score = 0
        total_turns = 0
        for turn in battle['battle_timeline']:

            p1_pokemon_state= turn.get('p1_pokemon_state', {}) # take the p1_pokemon_state dict
            p2_pokemon_state= turn.get('p2_pokemon_state', {})

            name1 = p1_pokemon_state.get('name') # take the name of the pokemon inside p1_pokemon_state dict
            name2 = p2_pokemon_state.get('name')

            p1_move_details = turn.get('p1_move_details', {})
            p2_move_details = turn.get('p2_move_details', {})

            if p1_move_details:
                p1_category = p1_move_details.get('category').upper()
                if p1_category == 'PHYSICAL':
                    p1_score += dict_base_stats[name1][0] / dict_base_stats[name2][1]  # base_atk / base_def
                elif p1_category == 'SPECIAL':
                    p1_score += dict_base_stats[name1][2] / dict_base_stats[name2][3]  # base_spa / base_spd
                elif p1_category == 'STATUS':
                    p1_score += 1 # neutral impact for status moves


            if p2_move_details:
                p2_category = p2_move_details.get('category').upper()
                if p2_category == 'PHYSICAL':
                    p2_score += dict_base_stats[name2][0] / dict_base_stats[name1][1]  # base_atk / base_def
                elif p2_category == 'SPECIAL':
                    p2_score += dict_base_stats[name2][2] / dict_base_stats[name1][3]  # base_spa / base_spd
                elif p2_category == 'STATUS':
                    p2_score += 1 # neutral impact for status moves
             
            total_turns += 1

        cat_impact_p1 = p1_score / total_turns 
        cat_impact_p2 = p2_score / total_turns

        result = {'battle_id': battle['battle_id']}

        if difference:
            result['cat_impact_diff'] = cat_impact_p1 - cat_impact_p2
        else: 
            result['p1_cat_impact_score'] = cat_impact_p1
            result['p2_cat_impact_score'] = cat_impact_p2
        
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
    return pd.DataFrame(final)


def avg_stab_multiplier(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    """
    Calculates the average STAB multiplier (1.5 for STAB, 1.0 for non-STAB/Status) 
    for all moves used by P1 and P2 throughout the battle.
    
    :param data: List of battle dictionaries.
    :param difference: If True, returns the difference (P1 - P2) in average STAB multipliers.
    :return: A pandas DataFrame with the calculated average STAB multiplier data.
    """
    # Helper dictionary to quickly get the attacking Pokémon's type(s)
    pokemon_att_types = get_dict_attacker_types(data)
    
    final = []
    
    for battle in data:
        total_turns = 0
        total_stab_p1 = 0.0
        total_stab_p2 = 0.0
        
        for turn in battle['battle_timeline']:
            p1_pokemon_state = turn.get('p1_pokemon_state', {})
            p2_pokemon_state = turn.get('p2_pokemon_state', {})

            name1 = p1_pokemon_state.get('name')
            name2 = p2_pokemon_state.get('name')

            # Get the types of the Pokémon currently active
            att_types_p1 = pokemon_att_types.get(name1, [])
            att_types_p2 = pokemon_att_types.get(name2, [])

            p1_move_details = turn.get('p1_move_details', {})
            p2_move_details = turn.get('p2_move_details', {})
            
            # --- P1 STAB Calculation ---
            if p1_move_details:
                move_type_1 = p1_move_details.get('type', '').lower()
                
                # Check if the move's type matches any of the attacker's types (STAB)
                if move_type_1 in att_types_p1:
                    total_stab_p1 += 1.5
                else:
                    total_stab_p1 += 1.0 # Non-STAB moves (including Status moves) get 1.0
            
            # --- P2 STAB Calculation ---
            if p2_move_details:
                move_type_2 = p2_move_details.get('type', '').lower()
                
                # Check if the move's type matches any of the attacker's types (STAB)
                if move_type_2 in att_types_p2:
                    total_stab_p2 += 1.5
                else:
                    total_stab_p2 += 1.0 # Non-STAB moves (including Status moves) get 1.0

            total_turns += 1
            
        # Handle division by zero
        
        avg_stab_p1 = total_stab_p1 / total_turns
        avg_stab_p2 = total_stab_p2 / total_turns

        result = {'battle_id': battle['battle_id']}
            
        if difference:
            result['avg_stab_diff'] = avg_stab_p1 - avg_stab_p2
        else: 
            result['avg_stab_p1'] = avg_stab_p1
            result['avg_stab_p2'] = avg_stab_p2
        
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
            
    return pd.DataFrame(final)


def avg_final_HP_pct(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    '''Calculate the average HP percentage of P1 and P2's Pokémon at the end of the 30 turns.'''

    final = []
    for battle in data:
        p1_dict={} # collect life percentages of each pokemon in p1 team
        p2_dict={} # collect life percentages of each pokemon in p2 team

        for turn in battle['battle_timeline']:
            p1_pokemon_state = turn.get('p1_pokemon_state', {})
            p2_pokemon_state = turn.get('p2_pokemon_state', {})

            name1 = p1_pokemon_state.get('name')
            name2 = p2_pokemon_state.get('name')

            Hp_1 = p1_pokemon_state.get('hp_pct')
            Hp_2 = p2_pokemon_state.get('hp_pct')

            p1_dict[name1] = Hp_1
            p2_dict[name2] = Hp_2

        # calculate the average even if a pokemon is fainted (0% hp) or not present (100% hp)
         # assuming a team of 6 pokemons
        avg_hp_pct_p1 = (np.sum(list(p1_dict.values())) + (6 - len(p1_dict)) ) / 6
        avg_hp_pct_p2 = (np.sum(list(p2_dict.values())) + (6 - len(p2_dict)) ) / 6 

        result = {'battle_id': battle['battle_id']}

        if difference:
            result['avg_final_hp_pct_diff'] = avg_hp_pct_p1 - avg_hp_pct_p2
        else:
            result['avg_final_hp_pct_p1'] = avg_hp_pct_p1
            result['avg_final_hp_pct_p2'] = avg_hp_pct_p2
        
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
    
    return pd.DataFrame(final)


def avg_stat_diff_per_turn(data: list[dict], stats: list[str], test: bool = False) -> pd.DataFrame:
    '''
    Calculate the average base stat difference (P1 - P2) per turn for multiple stats.
    
    Args:
        data: List of battle dictionaries
        stats: List of stat names to calculate ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
        
    Returns:
        DataFrame with battle_id, average stat differences per turn for each stat, and player_won
    '''
    # Get base stats dictionary for all pokemon
    pokemon_stats = get_dict_base_stats1(data)
    
    final = []
    for battle in data:
        # Initialize totals for each stat
        total_stat_diffs = {stat: 0.0 for stat in stats}
        total_turns = 0
        
        for turn in battle['battle_timeline']:
            p1_pokemon_state = turn.get('p1_pokemon_state', {})
            p2_pokemon_state = turn.get('p2_pokemon_state', {})

            name1 = p1_pokemon_state.get('name')
            name2 = p2_pokemon_state.get('name')
            
            # Get base stats for both pokemon
            stats_p1 = pokemon_stats.get(name1, {})
            stats_p2 = pokemon_stats.get(name2, {})
            
            # Calculate differences for each requested stat
            for stat in stats:
                if stat == 'hp':
                    # For HP, multiply by HP percentage
                    hp_pct_1 = p1_pokemon_state.get('hp_pct', 0)
                    hp_pct_2 = p2_pokemon_state.get('hp_pct', 0)
                    base_hp_1 = stats_p1.get('base_hp', 0)
                    base_hp_2 = stats_p2.get('base_hp', 0)
                    stat_1 = base_hp_1 * hp_pct_1
                    stat_2 = base_hp_2 * hp_pct_2
                else:
                    # For other stats, use base values directly
                    stat_1 = stats_p1.get(f'base_{stat}', 0)
                    stat_2 = stats_p2.get(f'base_{stat}', 0)
                
                # Calculate stat difference for this turn
                stat_diff = stat_1 - stat_2
                total_stat_diffs[stat] += stat_diff
            
            total_turns += 1
        
        # Calculate average stat differences per turn
        battle_result = {
            'battle_id': battle['battle_id']
        }
        
        for stat in stats:
            avg_stat_diff = total_stat_diffs[stat] / total_turns
            battle_result[f'avg_{stat}_diff_per_turn'] = avg_stat_diff
        
        if not test:
            battle_result['player_won'] = battle['player_won']
        
        final.append(battle_result)
    
    return pd.DataFrame(final)


def avg_boost_diff_per_turn(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    '''
    Calculate the average boost values for P1 and P2's Pokémon per turn.
    Boosts are temporary stat modifications that occur during battle.
    
    Args:
        data: List of battle dictionaries
        difference: If True, returns the difference (P1 - P2) in boost values
        
    Returns:
        DataFrame with battle_id, average boost values per turn for each boost type, and player_won
    '''
    # Common boost types in Pokémon battles
    boost_types = ['atk', 'def', 'spa', 'spd', 'spe']
    
    final = []
    for battle in data:
        # Initialize totals for each boost type
        total_boosts_p1 = {boost: 0.0 for boost in boost_types}
        total_boosts_p2 = {boost: 0.0 for boost in boost_types}
        total_turns = 0
        
        for turn in battle['battle_timeline']:
            p1_pokemon_state = turn.get('p1_pokemon_state', {})
            p2_pokemon_state = turn.get('p2_pokemon_state', {})
            
            # Get boost values for P1 pokemon
            p1_boosts = p1_pokemon_state.get('boosts', {})
            p2_boosts = p2_pokemon_state.get('boosts', {})
            
            # Sum up boosts for each type
            for boost_type in boost_types:
                p1_boost_value = p1_boosts.get(boost_type, 0)
                p2_boost_value = p2_boosts.get(boost_type, 0)
                
                total_boosts_p1[boost_type] += p1_boost_value
                total_boosts_p2[boost_type] += p2_boost_value
            
            total_turns += 1
        
        # Calculate average boosts per turn
        battle_result = {
            'battle_id': battle['battle_id']
        }
        
        # Calculate differences for each boost type
        for boost_type in boost_types:
            avg_boost_p1 = total_boosts_p1[boost_type] / total_turns if total_turns > 0 else 0.0
            avg_boost_p2 = total_boosts_p2[boost_type] / total_turns if total_turns > 0 else 0.0
            battle_result[f'avg_{boost_type}_boost_diff'] = avg_boost_p1 - avg_boost_p2
        
        if not test:
            battle_result['player_won'] = battle['player_won']
            
        final.append(battle_result)
    
    return pd.DataFrame(final)


def accuracy_basepower_avg(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    '''
    Calculate the average accuracy and base power of moves used by P1 and P2 throughout the battle.
    For turns where move_details is null (no move was made), accuracy and base_power are treated as 0.

    Args:
        data: List of battle dictionaries
        difference: If True, returns the difference (P1 - P2) in average accuracy, base power and priority
        test: If True, excludes the "player_won" column from the output.

    Returns:
        DataFrame with battle_id, average accuracy and base power for P1 and P2, and optionally player_won
    '''
    final = []
    for battle in data:
        total_turns = 0
        total_accuracy_p1 = 0.0
        total_accuracy_p2 = 0.0
        total_basepower_p1 = 0.0
        total_basepower_p2 = 0.0
        total_priority_p1 = 0.0
        total_priority_p2 = 0.0

        for turn in battle['battle_timeline']:
            p1_move_details = turn.get('p1_move_details', {})
            p2_move_details = turn.get('p2_move_details', {})

            # --- P1 Calculation ---
            if p1_move_details:
                accuracy_1 = p1_move_details.get('accuracy', 100)  # default accuracy is 100 if not specified
                base_power_1 = p1_move_details.get('base_power', 0)  # default base power is 0 if not specified
                priority_1 = p1_move_details.get('priority', 0)

                total_accuracy_p1 += accuracy_1
                total_basepower_p1 += base_power_1
                total_priority_p1 += priority_1
            else:
                total_accuracy_p1 += 0
                total_basepower_p1 += 0
                total_priority_p1 += 0

            # --- P2 Calculation ---
            if p2_move_details:
                accuracy_2 = p2_move_details.get('accuracy', 100)
                base_power_2 = p2_move_details.get('base_power', 0)
                priority_2 = p2_move_details.get('priority', 0)

                total_accuracy_p2 += accuracy_2
                total_basepower_p2 += base_power_2
                total_priority_p2 += priority_2
            else:
                total_accuracy_p2 += 0
                total_basepower_p2 += 0
                total_priority_p2 += 0

            total_turns += 1

        # Calculate averages
        avg_accuracy_p1 = total_accuracy_p1 / total_turns if total_turns > 0 else 0.0
        avg_accuracy_p2 = total_accuracy_p2 / total_turns if total_turns > 0 else 0.0
        avg_basepower_p1 = total_basepower_p1 / total_turns if total_turns > 0 else 0.0
        avg_basepower_p2 = total_basepower_p2 / total_turns if total_turns > 0 else 0.0
        avg_priority_p1 = total_priority_p1 / total_turns if total_turns > 0 else 0.0
        avg_priority_p2 = total_priority_p2 / total_turns if total_turns > 0 else 0.0

        result = {
            'battle_id': battle['battle_id']
        }

        if difference:
            result['avg_accuracy_diff'] = avg_accuracy_p1 - avg_accuracy_p2
            result['avg_basepower_diff'] = avg_basepower_p1 - avg_basepower_p2
            result['avg_priority_diff'] = avg_priority_p1 - avg_priority_p2

        else:
            result['avg_accuracy_p1'] = avg_accuracy_p1
            result['avg_accuracy_p2'] = avg_accuracy_p2
            result['avg_basepower_p1'] = avg_basepower_p1
            result['avg_basepower_p2'] = avg_basepower_p2
            result['avg_priority_p1'] = avg_priority_p1 
            result['avg_priority_p2'] = avg_priority_p2 

        if not test:
            result['player_won'] = battle['player_won']

        final.append(result)

    return pd.DataFrame(final)


def status_turn_diff(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    '''Count the total number of turns P1's Pokémon has a negative status (something different than nostatus) 
        and subtract the total turns P2's Pokémon has one.'''

    final = []
    for battle in data:
        total_status_turns_p1 = 0
        total_status_turns_p2 = 0

        for turn in battle['battle_timeline']:
            p1_pokemon_state = turn.get('p1_pokemon_state', {})
            p2_pokemon_state = turn.get('p2_pokemon_state', {})

            status_1 = p1_pokemon_state.get('status', 'nostatus').lower() 
            status_2 = p2_pokemon_state.get('status', 'nostatus').lower()

            if status_1 != 'nostatus':
                total_status_turns_p1 += 1

            if status_2 != 'nostatus':
                total_status_turns_p2 += 1

        result = {'battle_id': battle['battle_id']}

        if difference:
            result['status_turn_diff'] = total_status_turns_p1 - total_status_turns_p2
        else:
            result['total_status_turns_p1'] = total_status_turns_p1
            result['total_status_turns_p2'] = total_status_turns_p2

        if not test:
            result['player_won'] = battle['player_won']

        final.append(result)
    
    return pd.DataFrame(final)


def neg_effects_turn(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    '''Count the total number of turns P1's Pokémon has a negative volatile effect 
        and subtract the total turns P2's Pokémon has one.
    '''

    # This set is now tailored to your dataset
    NEGATIVE_EFFECTS = {
        'clamp', 'confusion', 'disable', 'firespin', 'wrap'
    }

    final = []
    for battle in data:
        total_neg_effects_turns_p1 = 0
        total_neg_effects_turns_p2 = 0

        for turn in battle['battle_timeline']:
            p1_pokemon_state = turn.get('p1_pokemon_state', {})
            p2_pokemon_state = turn.get('p2_pokemon_state', {})

            effects_1 = p1_pokemon_state.get('effects', ['noeffect'])
            effects_2 = p2_pokemon_state.get('effects', ['noeffect'])

            # Check if *any* of P1's current effects are in our negative list
            p1_has_neg_effect = any(effect in NEGATIVE_EFFECTS for effect in effects_1)
            if p1_has_neg_effect:
                total_neg_effects_turns_p1 += 1

            # Check if *any* of P2's current effects are in our negative list
            p2_has_neg_effect = any(effect in NEGATIVE_EFFECTS for effect in effects_2)
            if p2_has_neg_effect:
                total_neg_effects_turns_p2 += 1

        result = {'battle_id': battle['battle_id']}

        if difference:
            result['neg_effects_turn_diff'] = total_neg_effects_turns_p1 - total_neg_effects_turns_p2
        else:
            result['total_neg_effects_turns_p1'] = total_neg_effects_turns_p1
            result['total_neg_effects_turns_p2'] = total_neg_effects_turns_p2

        if not test:
            result['player_won'] = battle['player_won']

        final.append(result)
    
    return pd.DataFrame(final)


def avg_team_vs_lead_stats(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    """
    Calculates the average base stats for P1's team and compares them against
    the base stats of P2's lead Pokémon.
    This includes 'hp', 'atk', 'def', 'spa', 'spd', and 'spe'.

    Args:
        data (list): The list of battle dictionaries.
        difference (bool): If True, returns the difference (P1 Avg - P2 Lead) in stats.
        test (bool): If True, excludes the "player_won" column.

    Returns:
        A pandas DataFrame with the calculated stat comparisons.
    """
    final = []
    # List of all 6 base stats
    stats_to_calc = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']

    for battle in data:
        result = {'battle_id': battle['battle_id']}
        
        # Get P1's full team and P2's lead
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {}) # P2's lead is a single dict

        # --- Calculate P1 Mean Stats ---
        p1_mean_stats = {}
        for stat in stats_to_calc:
            key = f'base_{stat}'
            if p1_team: # Avoid division by zero if team is empty
                # Calculate mean for the stat, default to 0 if stat is missing
                p1_mean_stats[stat] = np.mean([pokemon.get(key, 0) for pokemon in p1_team])
            else:
                p1_mean_stats[stat] = 0.0

        # --- Get P2 Lead Stats ---
        p2_lead_stats = {}
        for stat in stats_to_calc:
            key = f'base_{stat}'
            # Get the stat directly from the lead's dict, default to 0 if missing
            p2_lead_stats[stat] = p2_lead.get(key, 0)

        # --- Populate the result dictionary based on 'difference' flag ---
        for stat in stats_to_calc:
            if difference:
                # Calculate the difference: P1_mean - P2_lead
                result[f'avg_team_vs_lead_{stat}_diff'] = p1_mean_stats[stat] - p2_lead_stats[stat]
            else:
                # Store P1 avg and P2 lead stats separately
                result[f'p1_avg_base_{stat}'] = p1_mean_stats[stat]
                result[f'p2_lead_base_{stat}'] = p2_lead_stats[stat]

        if not test:
            # Ensure 'player_won' is present before trying to access it
            if 'player_won' in battle:
                result['player_won'] = battle['player_won']
        
        final.append(result)

    return pd.DataFrame(final)


def faint_count_diff_extractor(data: list[dict], difference: bool = True, test: bool = False) -> pd.DataFrame:
    """
    Calculates the number of fainted Pokémon for both players.
    
    If difference is True (default):
        Returns the difference (P2 faints - P1 faints)
        A positive value means P2 lost more Pokémon (good for P1).
        A negative value means P1 lost more Pokémon (bad for P1).
    If difference is False:
        Returns separate counts for P1 and P2.
    """
    final_features = []
    
    for battle in data:
        battle_id = battle.get('battle_id')
        
        # Use sets to store the names of fainted Pokémon,
        # ensuring we only count each Pokémon once.
        p1_fainted_pokemon = set()
        p2_fainted_pokemon = set()
        
        for turn in battle.get('battle_timeline', []):
            # Check P1 state
            p1_state = turn.get('p1_pokemon_state', {})
            p1_name = p1_state.get('name')
            p1_hp_pct = p1_state.get('hp_pct')
            
            if p1_name is not None and p1_hp_pct == 0.0:
                p1_fainted_pokemon.add(p1_name)
                
            # Check P2 state
            p2_state = turn.get('p2_pokemon_state', {})
            p2_name = p2_state.get('name')
            p2_hp_pct = p2_state.get('hp_pct')
            
            if p2_name is not None and p2_hp_pct == 0.0:
                p2_fainted_pokemon.add(p2_name)
        
        battle_result = {'battle_id': battle_id}
        
        if difference:
            # Calculate the feature: P2 faints - P1 faints
            faint_diff = len(p2_fainted_pokemon) - len(p1_fainted_pokemon)
            battle_result['faint_count_diff'] = faint_diff
        else:
            battle_result['faint_count_p1'] = len(p1_fainted_pokemon)
            battle_result['faint_count_p2'] = len(p2_fainted_pokemon)
        
        if not test:
            # Include the target variable if it's not a test run
            battle_result['player_won'] = battle.get('player_won')
            
        final_features.append(battle_result)
        
    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(final_features)


def ratio_category_diff(data: list[dict], difference: bool = True, test=False):
    """ Calculate the proportions of move categories used throughout the 30 turns.
    If difference is True (default), returns the differences (p1-p2) of the proportions.
    If difference is False, returns separate proportions for each player.
    """

    final = []
    for battle in data:
        p1_phy = 0
        p1_spe = 0
        p1_sta = 0

        p2_phy = 0
        p2_spe = 0
        p2_sta = 0

        total_turns = 0
        for turn in battle['battle_timeline']:

            p1_move_details = turn.get('p1_move_details', {})
            p2_move_details = turn.get('p2_move_details', {})

            if p1_move_details:
                p1_category = p1_move_details.get('category').upper()
                if p1_category == 'PHYSICAL':
                    p1_phy +=1
                elif p1_category == 'SPECIAL':
                    p1_spe +=1
                elif p1_category == 'STATUS':
                    p1_sta +=1

            if p2_move_details:
                p2_category = p2_move_details.get('category').upper()
                if p2_category == 'PHYSICAL':
                    p2_phy +=1
                elif p2_category == 'SPECIAL':
                    p2_spe +=1
                elif p2_category == 'STATUS':
                    p2_sta +=1
             
            total_turns += 1

        p1_phy_ratio = p1_phy / total_turns
        p1_spe_ratio = p1_spe / total_turns
        p1_sta_ratio = p1_sta / total_turns

        p2_phy_ratio = p2_phy / total_turns
        p2_spe_ratio = p2_spe / total_turns
        p2_sta_ratio = p2_sta / total_turns

        result = {'battle_id': battle['battle_id']}

        if difference:
            result['phy_ratio_diff'] = p1_phy_ratio - p2_phy_ratio
            result['spe_ratio_diff'] = p1_spe_ratio - p2_spe_ratio
            result['sta_ratio_diff'] = p1_sta_ratio - p2_sta_ratio
        else:
            result['p1_phy_ratio'] = p1_phy_ratio
            result['p1_spe_ratio'] = p1_spe_ratio
            result['p1_sta_ratio'] = p1_sta_ratio
            result['p2_phy_ratio'] = p2_phy_ratio
            result['p2_spe_ratio'] = p2_spe_ratio
            result['p2_sta_ratio'] = p2_sta_ratio

        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
        
    return pd.DataFrame(final)


def calculate_voluntary_swap_diff(data: list[dict], difference: bool = True, test=False):
    """
    Calculates the number of voluntary swaps for both players.
    
    If difference is True (default):
        Returns the difference in the number of voluntary swaps (p1-p2).
    If difference is False:
        Returns separate counts for P1 and P2.
    
    This excludes forced replacements due to a Pokémon fainting.
    """

    final = []
    for battle in data:
        # Swap counters
        p1_swaps = 0
        p2_swaps = 0
        
        # Track last active Pokémon
        p1_last_pokemon = None
        p2_last_pokemon = None

        # Process only the first 30 turns
        timeline = battle.get('battle_timeline')
        
        # Skip empty or invalid battles
        if not timeline:
            continue 

        for turn in timeline:
            
            current_p1_state = turn.get('p1_pokemon_state', {})
            current_p2_state = turn.get('p2_pokemon_state', {})
            
            current_p1_name = current_p1_state.get('name')
            current_p2_name = current_p2_state.get('name')
            
            # Check if a move was used this turn
            p1_used_move = bool(turn.get('p1_move_details'))
            p2_used_move = bool(turn.get('p2_move_details'))


            
            # Handle the first turn (initial leads are not swaps)
            if p1_last_pokemon is None:
                p1_last_pokemon = current_p1_name
                p2_last_pokemon = current_p2_name
            else:
                p1_name_changed = current_p1_name != p1_last_pokemon
                p2_name_changed = current_p2_name != p2_last_pokemon


                # A voluntary swap is when the name changes AND no move is used.
                if p1_name_changed and not p1_used_move:
                    p1_swaps += 1
                
                if p2_name_changed and not p2_used_move:
                    p2_swaps += 1

                p1_last_pokemon = current_p1_name
                p2_last_pokemon = current_p2_name
             

        
        result = {'battle_id': battle['battle_id']}
        
        if difference:
            result['voluntary_swap_diff'] = p1_swaps - p2_swaps
        else:
            result['p1_voluntary_swaps'] = p1_swaps
            result['p2_voluntary_swaps'] = p2_swaps

        if not test:
            result['player_won'] = battle.get('player_won')
            
        final.append(result)
        
    return pd.DataFrame(final)


def hp_advantage_flip_count(data: list[dict], test: bool = False) -> pd.DataFrame:
    """
    Counts how many times the HP advantage "flips" between P1 and P2.
    A flip occurs when the player with the higher hp_pct changes
    from one turn to the next.
    
    This measures battle volatility.

    Args:
        data (list): The list of battle dictionaries.
        
        test (bool): If True, excludes the "player_won" column.

    Returns:
        A pandas DataFrame with the calculated flip counts.
    """
    final = []
    for battle in data:
        result = {'battle_id': battle['battle_id']}
        
        total_flips = 0
        p1_gained_adv_count = 0
        p2_gained_adv_count = 0
        
        # -1 if P2 has more HP, 0 if equal, 1 if P1 has more HP
        last_advantage_state = 0 

        for turn in battle['battle_timeline']:
            p1_state = turn.get('p1_pokemon_state', {})
            p2_state = turn.get('p2_pokemon_state', {})

            p1_hp = p1_state.get('hp_pct', 0.0)
            p2_hp = p2_state.get('hp_pct', 0.0)

            current_advantage_state = 0
            if p1_hp > p2_hp:
                current_advantage_state = 1
            elif p2_hp > p1_hp:
                current_advantage_state = -1

            # Check for a flip (and ignore state 0 -> 1 or 0 -> -1 at the start)
            if current_advantage_state != last_advantage_state and last_advantage_state != 0:
                total_flips += 1
                if current_advantage_state == 1: # P1 just gained the advantage
                    p1_gained_adv_count += 1
                elif current_advantage_state == -1: # P2 just gained the advantage
                    p2_gained_adv_count += 1

            # Update the last state *only if* it's not a tie
            if current_advantage_state != 0:
                last_advantage_state = current_advantage_state
        

        result['total_hp_adv_flips'] = total_flips

        if not test:
            result['player_won'] = battle.get('player_won')
            
        final.append(result)
        
    return pd.DataFrame(final)


def damage_efficiency_ratio(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    """
    Calculates the Damage Efficiency Ratio (DER) for both players.
    
    DER is defined as (Total Damage Dealt) / (Total Damage Taken).
    This feature tracks all HP loss for all known Pokémon on each team
    throughout the battle timeline.

    Handling Division by Zero:
    - If a player takes 0.0 damage, their DER is considered "infinite."
    - To represent this numerically, we set the DER to (Total Damage Dealt + 1.0).
    - This ensures that a player who dealt damage and took none has a higher
      score than a player who dealt no damage and took none (DER = 1.0).

    Args:
        data (list): The list of battle dictionaries.
        difference (bool): If True, returns the difference (P1_DER - P2_DER).
        test (bool): If True, excludes the "player_won" column.

    Returns:
        A pandas DataFrame with the calculated DER features.
    """
    final = []
    
    for battle in data:
        result = {'battle_id': battle['battle_id']}
        
        # Track total HP loss for each team.
        # total_damage_dealt_by_p1 == total_hp_loss_by_p2
        # total_damage_dealt_by_p2 == total_hp_loss_by_p1
        total_hp_loss_by_p1 = 0.0
        total_hp_loss_by_p2 = 0.0
        
        # Dictionaries to store the last known HP of each Pokémon
        p1_team_hp = {}
        p2_team_hp = {}

        # Initialize P1's team with 100% HP
        for pokemon in battle.get('p1_team_details', []):
            if pokemon.get('name'):
                p1_team_hp[pokemon.get('name')] = 1.0
        
        # Initialize P2's lead with 100% HP
        p2_lead = battle.get('p2_lead_details', {})
        if p2_lead.get('name'):
            p2_team_hp[p2_lead.get('name')] = 1.0

        # Iterate through the timeline to sum up all HP loss
        for turn in battle.get('battle_timeline', []):
            p1_state = turn.get('p1_pokemon_state', {})
            p2_state = turn.get('p2_pokemon_state', {})
            
            p1_name = p1_state.get('name')
            p2_name = p2_state.get('name')
            
            p1_hp_pct = p1_state.get('hp_pct')
            p2_hp_pct = p2_state.get('hp_pct')
            
            # Skip turns with incomplete data
            if p1_name is None or p2_name is None or p1_hp_pct is None or p2_hp_pct is None:
                continue

            # --- Calculate P1 HP Loss (Damage Dealt by P2) ---
            # Get last known HP, default to 1.0 for the first time seeing it
            last_p1_hp = p1_team_hp.get(p1_name, 1.0)
            
            if p1_hp_pct < last_p1_hp:
                # Add the difference (damage taken)
                total_hp_loss_by_p1 += (last_p1_hp - p1_hp_pct)
            
            # Update the last known HP for this Pokémon
            p1_team_hp[p1_name] = p1_hp_pct
            
            # --- Calculate P2 HP Loss (Damage Dealt by P1) ---
            # Get last known HP, default to 1.0 for the first time seeing it
            last_p2_hp = p2_team_hp.get(p2_name, 1.0)
            
            if p2_hp_pct < last_p2_hp:
                # Add the difference (damage taken)
                total_hp_loss_by_p2 += (last_p2_hp - p2_hp_pct)
            
            # Update the last known HP for this Pokémon
            p2_team_hp[p2_name] = p2_hp_pct

        # --- Calculate Final DERs ---
        
        # P1's DER: Damage P1 Dealt (P2 HP Loss) / Damage P1 Took (P1 HP Loss)
        if total_hp_loss_by_p1 == 0.0:
            p1_der = total_hp_loss_by_p2 + 1.0  # Handle division by zero
        else:
            p1_der = total_hp_loss_by_p2 / total_hp_loss_by_p1
            
        # P2's DER: Damage P2 Dealt (P1 HP Loss) / Damage P2 Took (P2 HP Loss)
        if total_hp_loss_by_p2 == 0.0:
            p2_der = total_hp_loss_by_p1 + 1.0  # Handle division by zero
        else:
            p2_der = total_hp_loss_by_p1 / total_hp_loss_by_p2

        # --- Format Output ---
        if difference:
            result['der_diff'] = p1_der - p2_der
        else:
            result['p1_der'] = p1_der
            result['p2_der'] = p2_der
        
        if not test:
            result['player_won'] = battle.get('player_won')
            
        final.append(result)
        
    return pd.DataFrame(final)


def pokemon_encoding(data: list[dict], one_hot: bool = False, test: bool = False) -> pd.DataFrame:
    """Encode pokemon presence or indices for P1 and P2.
    
    one_hot=True:
        creates columns p1_<pokemon> and p2_<pokemon> with 1/0 presence flags.
    one_hot=False:
        creates p1_pokemon_1..6 and p2_pokemon_1..6 with index in pokemon_list or -1 if missing/unknown.
        For P2, indices are assigned in order of appearance in the battle.
    """
    pokemon_list = [
        'alakazam', 'articuno', 'chansey', 'charizard', 'cloyster',
        'dragonite', 'exeggutor', 'gengar', 'golem', 'jolteon',
        'jynx', 'lapras', 'persian', 'rhydon', 'slowbro',
        'snorlax', 'starmie', 'tauros', 'victreebel', 'zapdos'
    ]
    name_to_idx = {n: i for i, n in enumerate(pokemon_list)}

    def team_names_from_details(team_details: list[dict]) -> list[str]:
        names = []
        for p in team_details or []:
            n = p.get('name')
            if n:
                names.append(n.lower())
        return names

    final = []
    for battle in data:
        row = {'battle_id': battle.get('battle_id')}
        # P1 team from details (expected present)
        p1_names = team_names_from_details(battle.get('p1_team_details', []))
        
        # P2 team: track Pokémon as they appear in battle
        p2_names = []
        # Add P2's lead Pokémon first
        lead_name = battle.get('p2_lead_details', {}).get('name')
        if lead_name:
            p2_names.append(lead_name.lower())
        
        # Then track any other Pokémon that appear in the timeline
        for turn in battle.get('battle_timeline', []):
            n = (turn.get('p2_pokemon_state') or {}).get('name')
            if n:
                nl = n.lower()
                if nl not in p2_names:  # Only add if not already seen
                    p2_names.append(nl)

        # Add encoding to the row based on the chosen method
        if one_hot:
            # One-hot encoding: 1 if pokemon is present, 0 if not
            for pname in pokemon_list:
                row[f'p1_{pname}'] = 1 if pname in p1_names else 0
                row[f'p2_{pname}'] = 1 if pname in p2_names else 0
        else:
            # Index encoding
            # P1: we know the full team, so encode all 6 slots
            for i in range(6):
                if i < len(p1_names):
                    row[f'p1_pokemon_{i+1}'] = name_to_idx.get(p1_names[i], -1)
                else:
                    row[f'p1_pokemon_{i+1}'] = -1
                    
            # P2: we only know observed Pokémon in order of appearance
            # We still create 6 slots for consistency, but most will be -1
            for i in range(6):
                if i < len(p2_names):
                    row[f'p2_pokemon_{i+1}'] = name_to_idx.get(p2_names[i], -1)
                else:
                    row[f'p2_pokemon_{i+1}'] = -1

        # Add the win/loss label if not in test mode
        if not test:
            row['player_won'] = battle.get('player_won')
            
        final.append(row)

    return pd.DataFrame(final)
    
