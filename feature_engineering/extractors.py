# feature_engineering/extractors.py
from collections import defaultdict
import numpy as np
import pandas as pd
from .utils import *

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
def avg_effectiveness2(data: list[dict], difference: bool = False, divide_turns: bool = True, test: bool = False) -> pd.DataFrame:
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
                battle_result['avg_effectiveness_diff'] = avg_eff_p1 - avg_eff_p2
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
                    battle_data[f'avg_effectiveness_diff_{segment_name}'] = avg_eff_p1 - avg_eff_p2
                else:
                    battle_data[f'avg_effectiveness_p1_{segment_name}'] = avg_eff_p1
                    battle_data[f'avg_effectiveness_p2_{segment_name}'] = avg_eff_p2
            
            if not test:
                battle_data['player_won'] = battle['player_won']
            
            final.append(battle_data)

    return pd.DataFrame(final)


def category_impact_score(data: list[dict], difference=False, divide_turns=True, test=False):
    """Given the database, calculate the average category impact score of all moves used by P1 and P2 throughout the turns.
       Category impact score is defined as: base_atk/base_def for Physical moves and base_spa/base_spd for Special moves.
       
       Args:
            data: List of battle dictionaries
            difference: If True, returns the difference between P1 and P2 scores
            divide_turns: If True, computes separate averages for first 10, middle 10, and last 10 turns
            test: If True, excludes player_won from output
    """
    
    TURN_SEGMENTS = {
        'first_10': (0, 10),
        'middle_10': (10, 20),
        'last_10': (-10, None)  # Use slicing logic: last 10 turns
    }

    dict_base_stats = get_dict_base_stats(data)
    final = []
    for battle in data:
        if not divide_turns:
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

                # Check if names exist in the dictionary before proceeding
                if name1 in dict_base_stats and name2 in dict_base_stats:
                    if p1_move_details:
                        p1_category = p1_move_details.get('category', '').upper()
                        if p1_category == 'PHYSICAL':
                            p1_atk = dict_base_stats[name1][0]
                            p2_def = dict_base_stats[name2][1]
                            p1_score += p1_atk / (p2_def if p2_def != 0 else 1)  # base_atk / base_def
                        elif p1_category == 'SPECIAL':
                            p1_spa = dict_base_stats[name1][2]
                            p2_spd = dict_base_stats[name2][3]
                            p1_score += p1_spa / (p2_spd if p2_spd != 0 else 1)  # base_spa / base_spd
                        elif p1_category == 'STATUS':
                            p1_score += 1 # neutral impact for status moves


                    if p2_move_details:
                        p2_category = p2_move_details.get('category', '').upper()
                        if p2_category == 'PHYSICAL':
                            p2_atk = dict_base_stats[name2][0]
                            p1_def = dict_base_stats[name1][1]
                            p2_score += p2_atk / (p1_def if p1_def != 0 else 1)  # base_atk / base_def
                        elif p2_category == 'SPECIAL':
                            p2_spa = dict_base_stats[name2][2]
                            p1_spd = dict_base_stats[name1][3]
                            p2_score += p2_spa / (p1_spd if p1_spd != 0 else 1)  # base_spa / base_spd
                        elif p2_category == 'STATUS':
                            p2_score += 1 # neutral impact for status moves
             
                total_turns += 1

            cat_impact_p1 = p1_score / total_turns if total_turns > 0 else 0.0
            cat_impact_p2 = p2_score / total_turns if total_turns > 0 else 0.0

            result = {'battle_id': battle['battle_id']}

            if difference:
                result['cat_impact_diff'] = cat_impact_p1 - cat_impact_p2
            else: 
                result['p1_cat_impact_score'] = cat_impact_p1
                result['p2_cat_impact_score'] = cat_impact_p2
            
            if not test:
                result['player_won'] = battle['player_won']
                
            final.append(result)
        
        else:  # divide_turns=True
            battle_data = {'battle_id': battle['battle_id']}
            timeline = battle['battle_timeline']
            
            for segment_name, (start, end) in TURN_SEGMENTS.items():
                # Determine the slice of the timeline for the current segment
                if start is not None and end is None and start < 0:
                    segment_timeline = timeline[start:]  # Last N turns
                elif start is not None and end is not None:
                    segment_timeline = timeline[start:end]  # Middle section
                else:
                    segment_timeline = []  # Corrected from 'timeline'

                # Initialize segment-specific counters
                p1_score = 0
                p2_score = 0
                segment_turns = len(segment_timeline)

                # Process the segment
                for turn in segment_timeline:
                    p1_pokemon_state = turn.get('p1_pokemon_state', {})
                    p2_pokemon_state = turn.get('p2_pokemon_state', {})
                    name1 = p1_pokemon_state.get('name')
                    name2 = p2_pokemon_state.get('name')

                    p1_move_details = turn.get('p1_move_details', {})
                    p2_move_details = turn.get('p2_move_details', {})
                    
                    # Check if names exist in the dictionary before proceeding
                    if name1 in dict_base_stats and name2 in dict_base_stats:
                        if p1_move_details:
                            p1_category = p1_move_details.get('category', '').upper()
                            if p1_category == 'PHYSICAL':
                                p1_atk = dict_base_stats[name1][0]
                                p2_def = dict_base_stats[name2][1]
                                p1_score += p1_atk / (p2_def if p2_def != 0 else 1)
                            elif p1_category == 'SPECIAL':
                                p1_spa = dict_base_stats[name1][2]
                                p2_spd = dict_base_stats[name2][3]
                                p1_score += p1_spa / (p2_spd if p2_spd != 0 else 1)
                            elif p1_category == 'STATUS':
                                p1_score += 1

                        if p2_move_details:
                            p2_category = p2_move_details.get('category', '').upper()
                            if p2_category == 'PHYSICAL':
                                p2_atk = dict_base_stats[name2][0]
                                p1_def = dict_base_stats[name1][1]
                                p2_score += p2_atk / (p1_def if p1_def != 0 else 1)
                            elif p2_category == 'SPECIAL':
                                p2_spa = dict_base_stats[name2][2]
                                p1_spd = dict_base_stats[name1][3]
                                p2_score += p2_spa / (p1_spd if p1_spd != 0 else 1)
                            elif p2_category == 'STATUS':
                                p2_score += 1

                # Calculate segment averages
                if segment_turns > 0:
                    cat_impact_p1 = p1_score / segment_turns
                    cat_impact_p2 = p2_score / segment_turns
                else:
                    cat_impact_p1 = 0.0
                    cat_impact_p2 = 0.0

                if difference:
                    battle_data[f'{segment_name}_cat_impact_diff'] = cat_impact_p1 - cat_impact_p2
                else:
                    battle_data[f'{segment_name}_p1_cat_impact'] = cat_impact_p1
                    battle_data[f'{segment_name}_p2_cat_impact'] = cat_impact_p2

            if not test:
                battle_data['player_won'] = battle['player_won']

            final.append(battle_data)
    
    return pd.DataFrame(final)


def avg_stab_multiplier(data: list[dict], difference: bool = False, divide_turns: bool = True, test: bool = False) -> pd.DataFrame:
    """
    Calculates the average STAB multiplier (1.5 for STAB, 1.0 for non-STAB/Status) 
    for all moves used by P1 and P2 throughout the battle.
    
    Args:
        data: List of battle dictionaries
        difference: If True, returns the difference (P1 - P2) in average STAB multipliers
        divide_turns: If True, computes separate averages for first 10, middle 10, and last 10 turns
        test: If True, excludes player_won from output
        
    Returns:
        A pandas DataFrame with the calculated average STAB multiplier data.
    """
    TURN_SEGMENTS = {
        'first_10': (0, 10),
        'middle_10': (10, 20),
        'last_10': (-10, None)  # Use slicing logic: last 10 turns
    }
    
    # Helper dictionary to quickly get the attacking Pokémon's type(s)
    pokemon_att_types = get_dict_attacker_types(data)
    
    final = []
    
    for battle in data:
        if not divide_turns:
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
                else:
                    # If no move, counts as 1.0
                    total_stab_p1 += 1.0
            
                # --- P2 STAB Calculation ---
                if p2_move_details:
                    move_type_2 = p2_move_details.get('type', '').lower()
                
                    # Check if the move's type matches any of the attacker's types (STAB)
                    if move_type_2 in att_types_p2:
                        total_stab_p2 += 1.5
                    else:
                        total_stab_p2 += 1.0 # Non-STAB moves (including Status moves) get 1.0
                else:
                    # If no move, counts as 1.0
                    total_stab_p2 += 1.0

                total_turns += 1
            
            # Handle division by zero
            avg_stab_p1 = total_stab_p1 / total_turns if total_turns > 0 else 0.0
            avg_stab_p2 = total_stab_p2 / total_turns if total_turns > 0 else 0.0

            result = {'battle_id': battle['battle_id']}
                
            if difference:
                result['avg_stab_diff'] = avg_stab_p1 - avg_stab_p2
            else: 
                result['avg_stab_p1'] = avg_stab_p1
                result['avg_stab_p2'] = avg_stab_p2
            
            if not test:
                result['player_won'] = battle['player_won']
                
            final.append(result)
        
        else:  # divide_turns=True
            battle_data = {'battle_id': battle['battle_id']}
            timeline = battle['battle_timeline']
            
            for segment_name, (start, end) in TURN_SEGMENTS.items():
                # Determine the slice of the timeline for the current segment
                if start is not None and end is None and start < 0:
                    segment_timeline = timeline[start:]  # Last N turns
                elif start is not None and end is not None:
                    segment_timeline = timeline[start:end]  # Middle section
                else:
                    segment_timeline = [] # Corrected from 'timeline'

                # Initialize segment-specific counters
                segment_turns = len(segment_timeline)
                segment_stab_p1 = 0.0
                segment_stab_p2 = 0.0

                # Process the segment
                for turn in segment_timeline:
                    p1_pokemon_state = turn.get('p1_pokemon_state', {})
                    p2_pokemon_state = turn.get('p2_pokemon_state', {})

                    name1 = p1_pokemon_state.get('name')
                    name2 = p2_pokemon_state.get('name')

                    # Get the types of the Pokémon currently active
                    att_types_p1 = pokemon_att_types.get(name1, [])
                    att_types_p2 = pokemon_att_types.get(name2, [])

                    p1_move_details = turn.get('p1_move_details', {})
                    p2_move_details = turn.get('p2_move_details', {})

                    # P1 STAB calculation
                    if p1_move_details:
                        move_type = p1_move_details.get('type', '').lower()
                        segment_stab_p1 += 1.5 if move_type in att_types_p1 else 1.0
                    else:
                        segment_stab_p1 += 1.0

                    # P2 STAB calculation
                    if p2_move_details:
                        move_type = p2_move_details.get('type', '').lower()
                        segment_stab_p2 += 1.5 if move_type in att_types_p2 else 1.0
                    else:
                        segment_stab_p2 += 1.0

                # Calculate segment averages
                if segment_turns > 0:
                    avg_stab_p1 = segment_stab_p1 / segment_turns
                    avg_stab_p2 = segment_stab_p2 / segment_turns
                else:
                    avg_stab_p1 = 0.0
                    avg_stab_p2 = 0.0

                if difference:
                    battle_data[f'{segment_name}_stab_diff'] = avg_stab_p1 - avg_stab_p2
                else:
                    battle_data[f'{segment_name}_stab_p1'] = avg_stab_p1
                    battle_data[f'{segment_name}_stab_p2'] = avg_stab_p2

            if not test:
                battle_data['player_won'] = battle['player_won']

            final.append(battle_data)
            
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
        full_hp_p1 = list(p1_dict.values()) + [1.0] * (6 - len(p1_dict))
        full_hp_p2 = list(p2_dict.values()) + [1.0] * (6 - len(p2_dict))

        avg_hp_pct_p1 = np.mean(full_hp_p1)
        avg_hp_pct_p2 = np.mean(full_hp_p2)

        var_hp_pct_p1 = np.var(full_hp_p1)
        var_hp_pct_p2 = np.var(full_hp_p1)

        result = {'battle_id': battle['battle_id']}

        if difference:
            result['avg_final_hp_pct_diff'] = avg_hp_pct_p1 - avg_hp_pct_p2
            result['var_final_hp_pct_diff'] = var_hp_pct_p1 - var_hp_pct_p2
        else:
            result['avg_final_hp_pct_p1'] = avg_hp_pct_p1
            result['avg_final_hp_pct_p2'] = avg_hp_pct_p2
            result['var_final_hp_pct_p1'] = var_hp_pct_p1
            result['var_final_hp_pct_p2'] = var_hp_pct_p2
        
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
    
    return pd.DataFrame(final)


def avg_stat_diff_per_turn(data: list[dict], stats: list[str], divide_turns: bool = True, test: bool = False) -> pd.DataFrame:
    '''
    Calculate the average base stat difference (P1 - P2) per turn for multiple stats.
    
    Args:
        data: List of battle dictionaries
        stats: List of stat names to calculate ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
        divide_turns: If True, computes separate averages for first 10, middle 10, and last 10 turns
        test: If True, excludes player_won from output
        
    Returns:
        DataFrame with battle_id, average stat differences per turn for each stat, and player_won
    '''
    TURN_SEGMENTS = {
        'first_10': (0, 10),
        'middle_10': (10, 20),
        'last_10': (-10, None)  # Use slicing logic: last 10 turns
    }
    
    # Get base stats dictionary for all pokemon
    pokemon_stats = get_dict_base_stats1(data)
    
    final = []
    for battle in data:
        if not divide_turns:
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
                avg_stat_diff = total_stat_diffs[stat] / total_turns if total_turns > 0 else 0.0
                battle_result[f'avg_{stat}_diff_per_turn'] = avg_stat_diff
            
            if not test:
                battle_result['player_won'] = battle['player_won']
            
            final.append(battle_result)
        
        else:  # divide_turns=True
            battle_data = {'battle_id': battle['battle_id']}
            timeline = battle['battle_timeline']
            
            for segment_name, (start, end) in TURN_SEGMENTS.items():
                # Determine the slice of the timeline for the current segment
                if start is not None and end is None and start < 0:
                    segment_timeline = timeline[start:]  # Last N turns
                elif start is not None and end is not None:
                    segment_timeline = timeline[start:end]  # Middle section
                else:
                    segment_timeline = [] # Corrected from 'timeline'

                # Initialize segment-specific counters
                segment_stat_diffs = {stat: 0.0 for stat in stats}
                segment_turns = len(segment_timeline)

                # Process the segment
                for turn in segment_timeline:
                    p1_pokemon_state = turn.get('p1_pokemon_state', {})
                    p2_pokemon_state = turn.get('p2_pokemon_state', {})

                    name1 = p1_pokemon_state.get('name')
                    name2 = p2_pokemon_state.get('name')
                    
                    # Get base stats for both pokemon
                    stats_p1 = pokemon_stats.get(name1, {})
                    stats_p2 = pokemon_stats.get(name2, {})
                    
                    # Calculate differences for each requested stat
                    for stat in stats:
                        # Corrected logic for HP (was missing in original 'else' block)
                        if stat == 'hp':
                            hp_pct_1 = p1_pokemon_state.get('hp_pct', 0)
                            hp_pct_2 = p2_pokemon_state.get('hp_pct', 0)
                            base_hp_1 = stats_p1.get('base_hp', 0)
                            base_hp_2 = stats_p2.get('base_hp', 0)
                            stat_1 = base_hp_1 * hp_pct_1
                            stat_2 = base_hp_2 * hp_pct_2
                        else:
                            stat_1 = stats_p1.get(f'base_{stat}', 0)
                            stat_2 = stats_p2.get(f'base_{stat}', 0)
                        
                        segment_stat_diffs[stat] += stat_1 - stat_2

                # Calculate segment averages
                if segment_turns > 0:
                    for stat in stats:
                        avg_stat_diff = segment_stat_diffs[stat] / segment_turns
                        battle_data[f'{segment_name}_{stat}_diff'] = avg_stat_diff
                else:
                    # Add default values if segment is empty
                    for stat in stats:
                        battle_data[f'{segment_name}_{stat}_diff'] = 0.0


            if not test:
                battle_data['player_won'] = battle['player_won']

            final.append(battle_data)
    
    return pd.DataFrame(final)


def avg_boost_diff_per_turn(data: list[dict], test: bool = False) -> pd.DataFrame:
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


def accuracy_avg(data: list[dict], difference: bool = False, divide_turns: bool = True, test: bool = False) -> pd.DataFrame:
    '''
    Calculate the average accuracy and base power of moves used by P1 and P2 throughout the battle.
    For turns where move_details is null (no move was made), accuracy and base_power are treated as 0.

    Args:
        data: List of battle dictionaries
        difference: If True, returns the difference (P1 - P2) in average accuracy and priority
        divide_turns: If True, computes separate averages for first 10, middle 10, and last 10 turns
        test: If True, excludes the "player_won" column from the output.

    Returns:
        DataFrame with battle_id, average accuracy and base power for P1 and P2, and optionally player_won
    '''
    TURN_SEGMENTS = {
        'first_10': (0, 10),
        'middle_10': (10, 20),
        'last_10': (-10, None)  # Use slicing logic: last 10 turns
    }
    
    final = []
    for battle in data:
        if not divide_turns:
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
                    # Use .get(key, default) to handle missing keys
                    accuracy_1 = p1_move_details.get('accuracy', 100)  # default accuracy is 100 if not specified
                    base_power_1 = p1_move_details.get('base_power', 0)  # default base power is 0 if not specified
                    priority_1 = p1_move_details.get('priority', 0)

                    total_accuracy_p1 += accuracy_1
                    total_basepower_p1 += base_power_1
                    total_priority_p1 += priority_1
                else:
                    # No move, add 0
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
                    # No move, add 0
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
                #result['avg_basepower_diff'] = avg_basepower_p1 - avg_basepower_p2
                result['avg_priority_diff'] = avg_priority_p1 - avg_priority_p2

            else:
                result['avg_accuracy_p1'] = avg_accuracy_p1
                result['avg_accuracy_p2'] = avg_accuracy_p2
                #result['avg_basepower_p1'] = avg_basepower_p1
                #result['avg_basepower_p2'] = avg_basepower_p2
                result['avg_priority_p1'] = avg_priority_p1 
                result['avg_priority_p2'] = avg_priority_p2 

            if not test:
                result['player_won'] = battle['player_won']

            final.append(result)
            
        else:  # divide_turns=True
            battle_data = {'battle_id': battle['battle_id']}
            timeline = battle['battle_timeline']
            
            for segment_name, (start, end) in TURN_SEGMENTS.items():
                # Determine the slice of the timeline for the current segment
                if start is not None and end is None and start < 0:
                    segment_timeline = timeline[start:]  # Last N turns
                elif start is not None and end is not None:
                    segment_timeline = timeline[start:end]  # Middle section
                else:
                    segment_timeline = [] # Corrected from 'timeline'

                # Initialize segment-specific counters
                segment_accuracy_p1 = 0.0
                segment_accuracy_p2 = 0.0
                segment_basepower_p1 = 0.0
                segment_basepower_p2 = 0.0
                segment_priority_p1 = 0.0
                segment_priority_p2 = 0.0
                segment_turns = len(segment_timeline)

                # Process the segment
                for turn in segment_timeline:
                    p1_move_details = turn.get('p1_move_details', {})
                    p2_move_details = turn.get('p2_move_details', {})

                    # P1 move details
                    if p1_move_details:
                        segment_accuracy_p1 += p1_move_details.get('accuracy', 100) # Use default 100
                        segment_basepower_p1 += p1_move_details.get('base_power', 0)
                        segment_priority_p1 += p1_move_details.get('priority', 0)
                    else:
                        segment_accuracy_p1 += 0
                        segment_basepower_p1 += 0
                        segment_priority_p1 += 0

                    # P2 move details
                    if p2_move_details:
                        segment_accuracy_p2 += p2_move_details.get('accuracy', 100) # Use default 100
                        segment_basepower_p2 += p2_move_details.get('base_power', 0)
                        segment_priority_p2 += p2_move_details.get('priority', 0)
                    else:
                        segment_accuracy_p2 += 0
                        segment_basepower_p2 += 0
                        segment_priority_p2 += 0

                # Calculate segment averages
                if segment_turns > 0:
                    avg_accuracy_p1 = segment_accuracy_p1 / segment_turns
                    avg_accuracy_p2 = segment_accuracy_p2 / segment_turns
                    avg_basepower_p1 = segment_basepower_p1 / segment_turns
                    avg_basepower_p2 = segment_basepower_p2 / segment_turns
                    avg_priority_p1 = segment_priority_p1 / segment_turns
                    avg_priority_p2 = segment_priority_p2 / segment_turns
                else:
                    avg_accuracy_p1 = 0.0
                    avg_accuracy_p2 = 0.0
                    avg_basepower_p1 = 0.0
                    avg_basepower_p2 = 0.0
                    avg_priority_p1 = 0.0
                    avg_priority_p2 = 0.0


                if difference:
                    battle_data[f'{segment_name}_accuracy_diff'] = avg_accuracy_p1 - avg_accuracy_p2
                    #battle_data[f'{segment_name}_basepower_diff'] = avg_basepower_p1 - avg_basepower_p2
                    battle_data[f'{segment_name}_priority_diff'] = avg_priority_p1 - avg_priority_p2
                else:
                    battle_data[f'{segment_name}_accuracy_p1'] = avg_accuracy_p1
                    battle_data[f'{segment_name}_accuracy_p2'] = avg_accuracy_p2
                    #battle_data[f'{segment_name}_basepower_p1'] = avg_basepower_p1
                    #battle_data[f'{segment_name}_basepower_p2'] = avg_basepower_p2
                    battle_data[f'{segment_name}_priority_p1'] = avg_priority_p1
                    battle_data[f'{segment_name}_priority_p2'] = avg_priority_p2

            if not test:
                battle_data['player_won'] = battle['player_won']

            final.append(battle_data)

    return pd.DataFrame(final)


def granular_turn_counts(data: list[dict], difference: bool = False, test: bool = False) -> pd.DataFrame:
    """
    Counts the total number of turns each player's Pokémon is afflicted
    with *each specific* status and negative volatile effect.
    """
    
    # Define all statuses and effects we care about
    # This ensures all columns are created, even if one doesn't appear
    ALL_STATUSES = ['slp', 'par', 'psn', 'brn', 'frz']
    NEGATIVE_EFFECTS = ['clamp', 'confusion', 'disable', 'firespin', 'wrap']

    final = []
    for battle in data:
        # Use defaultdict to automatically handle new keys with a 0 count
        p1_status_counts = defaultdict(int)
        p2_status_counts = defaultdict(int)
        p1_effect_counts = defaultdict(int)
        p2_effect_counts = defaultdict(int)

        for turn in battle['battle_timeline']:
            p1_state = turn.get('p1_pokemon_state', {})
            p2_state = turn.get('p2_pokemon_state', {})

            # --- Process Statuses ---
            status_1 = p1_state.get('status', 'nostatus').lower()
            if status_1 != 'nostatus':
                p1_status_counts[status_1] += 1

            status_2 = p2_state.get('status', 'nostatus').lower()
            if status_2 != 'nostatus':
                p2_status_counts[status_2] += 1

            # --- Process Volatile Effects ---
            effects_1 = p1_state.get('effects', [])
            for effect in effects_1:
                if effect in NEGATIVE_EFFECTS:
                    p1_effect_counts[effect] += 1
            
            effects_2 = p2_state.get('effects', [])
            for effect in effects_2:
                if effect in NEGATIVE_EFFECTS:
                    p2_effect_counts[effect] += 1

        # --- Build the result row ---
        result = {'battle_id': battle['battle_id']}

        # Add all status counts
        for status in ALL_STATUSES:
            p1_turns = p1_status_counts[status] # Gets 0 if status never appeared
            p2_turns = p2_status_counts[status]
            
            if difference:
                result[f'{status}_turn_diff'] = p1_turns - p2_turns
            else:
                result[f'p1_{status}_turns'] = p1_turns
                result[f'p2_{status}_turns'] = p2_turns
        
        # Add all negative effect counts
        for effect in NEGATIVE_EFFECTS:
            p1_turns = p1_effect_counts[effect]
            p2_turns = p2_effect_counts[effect]
            
            if difference:
                result[f'{effect}_turn_diff'] = p1_turns - p2_turns
            else:
                result[f'p1_{effect}_turns'] = p1_turns
                result[f'p2_{effect}_turns'] = p2_turns

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
        p2_lead = battle.get('p2_lead_details', {})  # P2's lead is a single dict
        
        # --- Calculate P1 Mean Stats ---
        p1_mean_stats = {}
        for stat in stats_to_calc:
            key = f'base_{stat}'
            if p1_team:  # Avoid division by zero if team is empty
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


def team_hp_advantage_flip_count(data: list[dict], team_size: int = 6, test: bool = False) -> pd.DataFrame:
    """
    Counts how many times the *average team HP* advantage "flips" between P1 and P2.
    
    This generalizes hp_advantage_flip_count by using the team-wide HP 
    calculation logic from avg_final_HP_pct (unseen Pokémon = 1.0 HP).
    
    A flip occurs when the player with the higher *average team hp_pct* changes from one turn to the next.

    This measures battle volatility at the team level.

    Args:
        data (list): The list of battle dictionaries.
        team_size (int): The number of Pokémon per team (default 6).
        test (bool): If True, excludes the "player_won" column.

    Returns:
        A pandas DataFrame with the calculated flip counts for each battle.
    """
    final = []
    for battle in data:
        result = {'battle_id': battle['battle_id']}
        
        total_flips = 0
        p1_gained_adv_count = 0
        p2_gained_adv_count = 0
        
        # -1 if P2 has more avg HP, 0 if equal, 1 if P1 has more avg HP
        last_advantage_state = 0 

        # These dicts store the last known HP for each Pokémon on a team
        # They are updated as the battle progresses
        p1_team_hp = {} # Key: pokemon name, Value: hp_pct
        p2_team_hp = {} # Key: pokemon name, Value: hp_pct

        for turn in battle['battle_timeline']:
            # 1. Update the known HP for the active Pokémon
            p1_state = turn.get('p1_pokemon_state', {})
            p2_state = turn.get('p2_pokemon_state', {})

            name1 = p1_state.get('name')
            name2 = p2_state.get('name')
            hp1 = p1_state.get('hp_pct')
            hp2 = p2_state.get('hp_pct')

            # Update dictionaries if the pokemon name and HP are valid
            if name1 and hp1 is not None:
                p1_team_hp[name1] = hp1
            if name2 and hp2 is not None:
                p2_team_hp[name2] = hp2

            # 2. Calculate current average team HP for P1
            # (sum of known HPs) + (1.0 * number of unseen Pokémon)
            p1_seen_count = len(p1_team_hp)
            p1_unseen_count = team_size - p1_seen_count
            p1_total_hp = np.sum(list(p1_team_hp.values())) + (p1_unseen_count * 1.0)
            avg_hp_pct_p1 = p1_total_hp / team_size

            # 3. Calculate current average team HP for P2
            p2_seen_count = len(p2_team_hp)
            p2_unseen_count = team_size - p2_seen_count
            p2_total_hp = np.sum(list(p2_team_hp.values())) + (p2_unseen_count * 1.0)
            avg_hp_pct_p2 = p2_total_hp / team_size

            # 4. Determine advantage state based on team averages
            current_advantage_state = 0
            if avg_hp_pct_p1 > avg_hp_pct_p2:
                current_advantage_state = 1
            elif avg_hp_pct_p2 > avg_hp_pct_p1:
                current_advantage_state = -1

            # 5. Check for a flip (and ignore state 0 -> 1 or 0 -> -1 at the start)
            if current_advantage_state != last_advantage_state and last_advantage_state != 0:
                total_flips += 1
                if current_advantage_state == 1: # P1 just gained the advantage
                    p1_gained_adv_count += 1
                elif current_advantage_state == -1: # P2 just gained the advantage
                    p2_gained_adv_count += 1

            # 6. Update the last state *only if* it's not a tie
            if current_advantage_state != 0:
                last_advantage_state = current_advantage_state
        
        # Store results for the battle
        result['total_team_hp_adv_flips'] = total_flips
        result['p1_gained_team_adv_count'] = p1_gained_adv_count
        result['p2_gained_team_adv_count'] = p2_gained_adv_count

        if not test:
            result['player_won'] = battle.get('player_won')
            
        final.append(result)
        
    return pd.DataFrame(final)


def damage_efficiency_ratio(data: list[dict], difference: bool = False, divide_turns: bool = True, test: bool = False) -> pd.DataFrame:
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
        divide_turns (bool): If True, computes separate ratios for first 10, middle 10, and last 10 turns.
        test (bool): If True, excludes the "player_won" column.

    Returns:
        A pandas DataFrame with the calculated DER features.
    """
    TURN_SEGMENTS = {
        'first_10': (0, 10),
        'middle_10': (10, 20),
        'last_10': (-10, None) # Use slicing logic: last 10 turns
    }
    
    final = []
    
    for battle in data:
        result = {'battle_id': battle['battle_id']}
        timeline = battle.get('battle_timeline', [])

        if not divide_turns:
            # Track total HP loss for each team.
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
            for turn in timeline:
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
                last_p1_hp = p1_team_hp.get(p1_name, 1.0)
                if p1_hp_pct < last_p1_hp:
                    total_hp_loss_by_p1 += (last_p1_hp - p1_hp_pct)
                p1_team_hp[p1_name] = p1_hp_pct
                
                # --- Calculate P2 HP Loss (Damage Dealt by P1) ---
                last_p2_hp = p2_team_hp.get(p2_name, 1.0)
                if p2_hp_pct < last_p2_hp:
                    total_hp_loss_by_p2 += (last_p2_hp - p2_hp_pct)
                p2_team_hp[p2_name] = p2_hp_pct

            # Calculate DERs
            if total_hp_loss_by_p1 == 0.0:
                p1_der = total_hp_loss_by_p2 + 1.0
            else:
                p1_der = total_hp_loss_by_p2 / total_hp_loss_by_p1
                
            if total_hp_loss_by_p2 == 0.0:
                p2_der = total_hp_loss_by_p1 + 1.0
            else:
                p2_der = total_hp_loss_by_p1 / total_hp_loss_by_p2

            if difference:
                result['der_diff'] = p1_der - p2_der
            else:
                result['p1_der'] = p1_der
                result['p2_der'] = p2_der

        else:  # divide_turns=True
            for segment_name, (start, end) in TURN_SEGMENTS.items():
                # Determine the slice of the timeline for the current segment
                if start is not None and end is None and start < 0:
                    segment_timeline = timeline[start:]  # Last N turns
                elif start is not None and end is not None:
                    segment_timeline = timeline[start:end]  # Middle section
                else:
                    segment_timeline = []

                # Initialize segment-specific tracking
                segment_hp_loss_by_p1 = 0.0
                segment_hp_loss_by_p2 = 0.0
                
                p1_team_hp = {}
                p2_team_hp = {}
                
                # Initialize P1's team
                for pokemon in battle.get('p1_team_details', []):
                    if pokemon.get('name'):
                        p1_team_hp[pokemon.get('name')] = 1.0
                
                # Initialize P2's lead
                p2_lead = battle.get('p2_lead_details', {})
                if p2_lead.get('name'):
                    p2_team_hp[p2_lead.get('name')] = 1.0

                # Process the segment
                for turn in segment_timeline:
                    p1_state = turn.get('p1_pokemon_state', {})
                    p2_state = turn.get('p2_pokemon_state', {})
                    
                    p1_name = p1_state.get('name')
                    p2_name = p2_state.get('name')
                    
                    p1_hp_pct = p1_state.get('hp_pct')
                    p2_hp_pct = p2_state.get('hp_pct')
                    
                    if p1_name is None or p2_name is None or p1_hp_pct is None or p2_hp_pct is None:
                        continue

                    # Calculate P1 HP Loss
                    last_p1_hp = p1_team_hp.get(p1_name, 1.0)
                    if p1_hp_pct < last_p1_hp:
                        segment_hp_loss_by_p1 += (last_p1_hp - p1_hp_pct)
                    p1_team_hp[p1_name] = p1_hp_pct
                    
                    # Calculate P2 HP Loss
                    last_p2_hp = p2_team_hp.get(p2_name, 1.0)
                    if p2_hp_pct < last_p2_hp:
                        segment_hp_loss_by_p2 += (last_p2_hp - p2_hp_pct)
                    p2_team_hp[p2_name] = p2_hp_pct

                # Calculate segment DERs
                if segment_hp_loss_by_p1 == 0.0:
                    p1_der = segment_hp_loss_by_p2 + 1.0
                else:
                    p1_der = segment_hp_loss_by_p2 / segment_hp_loss_by_p1
                    
                if segment_hp_loss_by_p2 == 0.0:
                    p2_der = segment_hp_loss_by_p1 + 1.0
                else:
                    p2_der = segment_hp_loss_by_p1 / segment_hp_loss_by_p2

                # Store segment results
                if difference:
                    result[f'{segment_name}_der_diff'] = p1_der - p2_der
                else:
                    result[f'{segment_name}_p1_der'] = p1_der
                    result[f'{segment_name}_p2_der'] = p2_der

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

    final = []
    for battle in data:
        row = {'battle_id': battle.get('battle_id')}
        
        # Initialize empty lists for both players
        p1_names = []
        p2_names = []
        
        # Track Pokemon only as they appear in the timeline for both players
        for turn in battle.get('battle_timeline', []):
            # Track P1's Pokemon
            n1 = (turn.get('p1_pokemon_state') or {}).get('name')
            if n1:
                nl1 = n1.lower()
                if nl1 not in p1_names:  # Only add if not already seen
                    p1_names.append(nl1)
                    
            # Track P2's Pokemon
            n2 = (turn.get('p2_pokemon_state') or {}).get('name')
            if n2:
                nl2 = n2.lower()
                if nl2 not in p2_names:  # Only add if not already seen
                    p2_names.append(nl2)

        # Track fainted status for each Pokemon
        p1_fainted = set()
        p2_fainted = set()
        for turn in battle.get('battle_timeline', []):
            p1_state = turn.get('p1_pokemon_state', {})
            p2_state = turn.get('p2_pokemon_state', {})
            
            # Check P1 Pokemon
            p1_name = p1_state.get('name')
            if p1_name:
                p1_name = p1_name.lower()
                if p1_state.get('hp_pct') == 0.0:
                    p1_fainted.add(p1_name)
            
            # Check P2 Pokemon
            p2_name = p2_state.get('name')
            if p2_name:
                p2_name = p2_name.lower()
                if p2_state.get('hp_pct') == 0.0:
                    p2_fainted.add(p2_name)

        # Add encoding to the row based on the chosen method
        if one_hot:
            # One-hot encoding: separate columns for seen and fainted
            for pname in pokemon_list:
                # P1 encoding
                row[f'p1_{pname}_seen'] = 1 if pname in p1_names else 0
                row[f'p1_{pname}_fainted'] = 1 if pname in p1_fainted else 0
                
                # P2 encoding
                row[f'p2_{pname}_seen'] = 1 if pname in p2_names else 0
                row[f'p2_{pname}_fainted'] = 1 if pname in p2_fainted else 0
        else:
            # Index encoding with status
            # P1: encode Pokemon index and status
            for i in range(6):
                if i < len(p1_names):
                    pokemon_name = p1_names[i]
                    row[f'p1_pokemon_{i+1}'] = name_to_idx.get(pokemon_name, -1)
                    row[f'p1_status_{i+1}'] = 0 if pokemon_name in p1_fainted else 1
                else:
                    row[f'p1_pokemon_{i+1}'] = -1
                    row[f'p1_status_{i+1}'] = -1  # not seen in timeline
                    
            # P2: encode Pokemon index and status
            for i in range(6):
                if i < len(p2_names):
                    pokemon_name = p2_names[i]
                    row[f'p2_pokemon_{i+1}'] = name_to_idx.get(pokemon_name, -1)
                    row[f'p2_status_{i+1}'] = 0 if pokemon_name in p2_fainted else 1
                else:
                    row[f'p2_pokemon_{i+1}'] = -1
                    row[f'p2_status_{i+1}'] = -1  # not seen in timeline

        # Add the win/loss label if not in test mode
        if not test:
            row['player_won'] = battle.get('player_won')
            
        final.append(row)

    return pd.DataFrame(final)


def avg_approx_damage(data: list[dict], difference: bool = True, test: bool = False) -> pd.DataFrame:
    """
    Calculates the average approximate damage dealt by P1 and P2 per turn
    over the entire battle, based on the Gen 1 damage formula.
    
    Formula: ( ( (2*Lvl)/5 +2) * Atk/Def * Base_power)/50 + 2 ) * Modifier
    
    Approximations made:
    - Level (Lvl) is 100. Constant = (2*100)/5 + 2 = 42
    - Atk/Def stats are Base Stats (not boosted).
    - Modifier = STAB * Type * random_avg * crt
    - STAB is 1.5 if applicable, 1.0 otherwise.
    - Type is the type effectiveness multiplier.
    - random_avg is (217..255)/255 = 0.925
    - Critical hits (crt) treated as 1.0
    - Status moves deal 0.0 damage.
    
    Args:
        data: List of battle dictionaries
        difference: If True, returns the difference (P1 - P2) in average damage
        test: If True, excludes player_won from output
    """
    
    # Constants from the damage formula
    LEVEL_CONSTANT = 42.0  # ( (2 * 100) / 5 + 2 )
    RANDOM_AVG = 0.925     # Average of (217..255)/255

    # Get helper dictionaries
    dict_base_stats = get_dict_base_stats(data)
    pokemon_att_types = get_dict_attacker_types(data)
    pokemon_def_types = get_dict_def_types(data)
    
    final = []
    
    for battle in data:
        # Calculate damage for entire battle
        total_turns = 0
        total_approx_damage_p1 = 0.0
        total_approx_damage_p2 = 0.0
        
        # Process all turns
        for turn in battle['battle_timeline']:
            p1_pokemon_state = turn.get('p1_pokemon_state', {})
            p2_pokemon_state = turn.get('p2_pokemon_state', {})

            name1 = p1_pokemon_state.get('name')
            name2 = p2_pokemon_state.get('name')

            p1_move_details = turn.get('p1_move_details', {})
            p2_move_details = turn.get('p2_move_details', {})
            
            # Check if we have stat data for both Pokemon
            if name1 in dict_base_stats and name2 in dict_base_stats:
                stats_p1 = dict_base_stats[name1] # [atk, def, spa, spd]
                stats_p2 = dict_base_stats[name2]
                
                att_types_p1 = pokemon_att_types.get(name1, [])
                def_types_p1 = pokemon_def_types.get(name1)
                
                att_types_p2 = pokemon_att_types.get(name2, [])
                def_types_p2 = pokemon_def_types.get(name2)

                # --- P1 Damage Calculation ---
                if p1_move_details:
                    category_1 = p1_move_details.get('category', '').upper()
                    base_power_1 = p1_move_details.get('base_power', 0)
                    move_type_1 = p1_move_details.get('type')

                    if category_1 in ('PHYSICAL', 'SPECIAL') and base_power_1 > 0 and move_type_1 and def_types_p2:
                        # 1. Get Stat Ratio
                        if category_1 == 'PHYSICAL':
                            p1_attack = stats_p1[0] # atk
                            p2_defense = stats_p2[1] # def
                            stat_ratio_1 = p1_attack / (p2_defense if p2_defense != 0 else 1)
                        else: # SPECIAL
                            p1_attack = stats_p1[2] # spa
                            p2_defense = stats_p2[3] # spd
                            stat_ratio_1 = p1_attack / (p2_defense if p2_defense != 0 else 1)
                        
                        # 2. Get Modifier
                        stab_1 = 1.5 if move_type_1.lower() in att_types_p1 else 1.0
                        type_eff_1 = effectiveness(move_type_1, def_types_p2)
                        modifier_1 = stab_1 * type_eff_1 * RANDOM_AVG
                        
                        # 3. Calculate Damage
                        damage_1 = (((LEVEL_CONSTANT * stat_ratio_1 * base_power_1) / 50) + 2) * modifier_1
                        total_approx_damage_p1 += damage_1

                # --- P2 Damage Calculation ---
                if p2_move_details:
                    category_2 = p2_move_details.get('category', '').upper()
                    base_power_2 = p2_move_details.get('base_power', 0)
                    move_type_2 = p2_move_details.get('type')
                    
                    if category_2 in ('PHYSICAL', 'SPECIAL') and base_power_2 > 0 and move_type_2 and def_types_p1:
                        # 1. Get Stat Ratio
                        if category_2 == 'PHYSICAL':
                            p2_attack = stats_p2[0] # atk
                            p1_defense = stats_p1[1] # def
                            stat_ratio_2 = p2_attack / (p1_defense if p1_defense != 0 else 1)
                        else: # SPECIAL
                            p2_attack = stats_p2[2] # spa
                            p1_defense = stats_p1[3] # spd
                            stat_ratio_2 = p2_attack / (p1_defense if p1_defense != 0 else 1)
                        
                        # 2. Get Modifier
                        stab_2 = 1.5 if move_type_2.lower() in att_types_p2 else 1.0
                        type_eff_2 = effectiveness(move_type_2, def_types_p1)
                        modifier_2 = stab_2 * type_eff_2 * RANDOM_AVG
                        
                        # 3. Calculate Damage
                        damage_2 = (((LEVEL_CONSTANT * stat_ratio_2 * base_power_2) / 50) + 2) * modifier_2
                        total_approx_damage_p2 += damage_2
            
            total_turns += 1

        # Calculate averages for entire battle
        avg_damage_p1 = total_approx_damage_p1 / total_turns if total_turns > 0 else 0.0
        avg_damage_p2 = total_approx_damage_p2 / total_turns if total_turns > 0 else 0.0

        result = {'battle_id': battle['battle_id']}

        if difference:
            result['avg_approx_damage_diff'] = avg_damage_p1 - avg_damage_p2
        else: 
            result['p1_avg_approx_damage'] = avg_damage_p1
            result['p2_avg_approx_damage'] = avg_damage_p2
        
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)

    return pd.DataFrame(final)


def first_KO_momentum_feature(data: list[dict], test: bool = False) -> pd.DataFrame:
    """
    Compute a single numeric feature per battle:
      +1/turn if player (P1) scores the first KO,
      -1/turn if opponent (P2) scores the first KO,
       0 if no KO occurs within the 30 recorded turns.
    """

    final = []
    for battle in data:
        timeline = battle.get("battle_timeline", [])
        first_KO_turn = 31   # sentinel for "no KO in first 30 turns"
        first_KO_side = None

        prev_p1_hp = 1.0
        prev_p2_hp = 1.0

        for turn in timeline:
            p1_hp = turn.get("p1_pokemon_state", {}).get("hp_pct", 1.0)
            p2_hp = turn.get("p2_pokemon_state", {}).get("hp_pct", 1.0)

            # detect a KO (HP drops from >0 to 0)
            if prev_p2_hp > 0 and p2_hp == 0:
                first_KO_turn = turn.get("turn", 31)
                first_KO_side = "p1"
                break
            elif prev_p1_hp > 0 and p1_hp == 0:
                first_KO_turn = turn.get("turn", 31)
                first_KO_side = "p2"
                break

            prev_p1_hp, prev_p2_hp = p1_hp, p2_hp

        # Encode into a single numeric value
        if first_KO_side == "p1":
            feature_value = 1.0 / first_KO_turn
        elif first_KO_side == "p2":
            feature_value = -1.0 / first_KO_turn
        else:
            feature_value = 0.0

        result = {
            "battle_id": battle["battle_id"],
            "first_KO_momentum": feature_value
        }

        if not test:
            result["player_won"] = battle.get("player_won")

        final.append(result)

    return pd.DataFrame(final)


def last_turn_status_extractor(data: list[dict], test: bool = False) -> pd.DataFrame:
    """
    Calculates the difference in active Pokémon status on the final turn.
    Relies directly on the 'status' field from the data.

    Args:
        data (list): The list of battle dictionaries.
        test (bool): If True, excludes the "player_won" column.

    Returns:
        A pandas DataFrame with 'status_<status>_diff' columns.
    """
    STATUSES_TO_CHECK = {'brn', 'fnt', 'frz', 'nostatus', 'par', 'psn', 'slp', 'tox'}

    final = []
    
    for battle in data:
        result = {'battle_id': battle['battle_id']}
        
        # Get the state data from the very last turn
        last_turn = battle["battle_timeline"][-1]
        p1_state = last_turn["p1_pokemon_state"]
        p2_state = last_turn["p2_pokemon_state"]

        # Determine effective status for P1 (directly from data)
        p1_effective_status = p1_state["status"]
        
        # Determine effective status for P2 (directly from data)
        p2_effective_status = p2_state["status"]
        
        # Calculate the diff for each status
        for status in STATUSES_TO_CHECK:
            p1_count = int(p1_effective_status == status)
            p2_count = int(p2_effective_status == status)
            result[f"status_{status}_diff"] = p1_count - p2_count
        
        if not test:
            result['player_won'] = battle['player_won']
        
        final.append(result)
        
    return pd.DataFrame(final)


def tot_pok_used(data: list[dict], test: bool = False) -> pd.DataFrame:

    final = []
    
    for battle in data:
        result = {'battle_id': battle['battle_id']}

        teams=get_last_hp(battle=battle,include_fainted=True)

        team1= teams.get('observed_P1')
        team2= teams.get('observed_P2')

        result['pok_used_diff'] = len(team1)- len(team2)
        
        if not test:
            result['player_won'] = battle['player_won']
        
        final.append(result)
        
    return pd.DataFrame(final)


def final_type_advantage(data: list[dict], difference: bool = True, test: bool = False) -> pd.DataFrame:

    pokemon_Stab_types = get_dict_def_types(data)

    final = []

    for battle in data:

        pokemon_dict_1 = {}
        pokemon_dict_2 = {}
        for turn in battle['battle_timeline']:
            p1_pokemon_state= turn.get('p1_pokemon_state') # take the p1_pokemon_state dict
            p2_pokemon_state= turn.get('p2_pokemon_state')

            name1 = p1_pokemon_state.get('name') # take the name of the pokemon insidide p1_pokemon_state dict
            name2 = p2_pokemon_state.get('name')

            hp_pct_1 = p1_pokemon_state.get('hp_pct')
            hp_pct_2 = p2_pokemon_state.get('hp_pct')

            type_stab_1 = list(pokemon_Stab_types.get(name1))
            type_stab_2 = list(pokemon_Stab_types.get(name2))

            pokemon_dict_1[name1] = [hp_pct_1,type_stab_1]
            pokemon_dict_2[name2] = [hp_pct_2,type_stab_2]


        survived_p1 = {k: v for k, v in pokemon_dict_1.items() if v[0] > 0}
        survived_p2 = {k: v for k, v in pokemon_dict_2.items() if v[0] > 0}
     
        # now they contain only the survived pokemons among the observed in the first 30 turns

        # compute power of P1 against P2
        power_p1 = 0
        for pok1 in survived_p1:
            for tp1 in survived_p1.get(pok1)[1]:
                if tp1 != 'notype':
                    for pok2 in survived_p2:
                        tps2 = survived_p2.get(pok2)[1]
                        power_p1 += effectiveness(tp1, tps2) # Assumes effectiveness(type, [list_of_types])

         # compute power of P2 against P1
        power_p2 = 0
        for pok2 in survived_p2:
            # <--- FIX 1: Corrected 'pok1' to 'pok2'
            for tp2 in survived_p2.get(pok2)[1]:
                if tp2 != 'notype':
                    for pok1 in survived_p1:
                        tps1 = survived_p1.get(pok1)[1]
                        # <--- FIX 2: Corrected 'power_p1' to 'power_p2'
                        power_p2 += effectiveness(tp2, tps1)


        if len(survived_p1) > 0 and len(survived_p2) > 0:
            power_p1 /= len(survived_p1) * len(survived_p2)
            power_p2 /= len(survived_p2) * len(survived_p1)
        else:
            # If one side has no survivors, set the powers sensibly
            # Example: if no survivors, that side’s power is 0
            if len(survived_p1) == 0:
                power_p1 = 0
            if len(survived_p2) == 0:
                power_p2 = 0



        result = {'battle_id': battle['battle_id']}

        if difference:
            result['final_type_advantage_diff'] = power_p1 - power_p2
            
        else:
            result['final_type_advantage_p1'] = power_p1 
            result['final_type_advantage_p2'] = power_p2
        if not test:
            result['player_won'] = battle['player_won']
            
        final.append(result)
    
    return pd.DataFrame(final)