# feature_engineering/extractors.py

import numpy as np
import pandas as pd
from .utils import get_dict_def_types, get_dict_attacker_types, get_dict_base_stats, effectiveness

def avg_effectiveness_1(data: list[dict], difference=False) -> pd.DataFrame:
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
        if difference:
            final.append({
                'battle_id': battle['battle_id'],
                'avg_effectiveness_diff': avg_eff_p1-avg_eff_p2,
                "player_won" : battle["player_won"]
            })
        else: 
            final.append({
                'battle_id': battle['battle_id'],
                'avg_effectiveness_p1': avg_eff_p1,
                'avg_effectiveness_p2': avg_eff_p2,
                "player_won" : battle["player_won"]
            })
    return pd.DataFrame(final)


def avg_effectiveness_1_1(data: list[dict], difference=False, include_status_moves=True) -> pd.DataFrame:
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
            
        if difference:
            final.append({
                'battle_id': battle['battle_id'],
                'avg_effectiveness_diff': avg_eff_p1-avg_eff_p2,
                "player_won" : battle["player_won"]
            })
        else: 
            final.append({
                'battle_id': battle['battle_id'],
                'avg_effectiveness_p1': avg_eff_p1,
                'avg_effectiveness_p2': avg_eff_p2,
                "player_won" : battle["player_won"]
            })
    return pd.DataFrame(final)

# Extended version of avg_effectiveness with turn segmentation
def avg_effectiveness2(data: list[dict], difference: bool = False, divide_turns: bool = False) -> pd.DataFrame:
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
            
            battle_result = {'battle_id': battle['battle_id'], "player_won": battle["player_won"]}
            if difference:
                battle_result['avg_diff'] = avg_eff_p1 - avg_eff_p2
            else: 
                battle_result['avg_effectiveness_p1'] = avg_eff_p1
                battle_result['avg_effectiveness_p2'] = avg_eff_p2
            
            final.append(battle_result)

        # --- Logic for Divided Turns (`divide_turns=True`) ---
        else:
            battle_data = {'battle_id': battle['battle_id'], "player_won": battle["player_won"]}
            
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
            
            final.append(battle_data)

    return pd.DataFrame(final)


def category_impact_score(data: list[dict], difference=False):

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

        if difference:
            final.append({
                'battle_id': battle['battle_id'],
                'cat_impact_diff': cat_impact_p1 - cat_impact_p2,
                "player_won" : battle["player_won"]
            })
        else: 
            final.append({
                'battle_id': battle['battle_id'],
                'p1_cat_impact_score': cat_impact_p1,
                'p2_cat_impact_score': cat_impact_p2,
                "player_won" : battle["player_won"]
            })
    return pd.DataFrame(final)


def avg_stab_multiplier(data: list[dict], difference: bool = False) -> pd.DataFrame:
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
            
        if difference:
            final.append({
                'battle_id': battle['battle_id'],
                'avg_stab_diff': avg_stab_p1 - avg_stab_p2,
                "player_won" : battle["player_won"]
            })
        else: 
            final.append({
                'battle_id': battle['battle_id'],
                'avg_stab_p1': avg_stab_p1,
                'avg_stab_p2': avg_stab_p2,
                "player_won" : battle["player_won"]
            })
            
    return pd.DataFrame(final)


def avg_final_HP_pct(data: list[dict], difference: bool = False) -> pd.DataFrame:
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

        if difference:
            final.append({
                'battle_id': battle['battle_id'],
                'avg_final_hp_pct_diff': avg_hp_pct_p1 - avg_hp_pct_p2,
                "player_won" : battle["player_won"]
            })
        else:
            final.append({
                'battle_id': battle['battle_id'],
                'avg_final_hp_pct_p1': avg_hp_pct_p1,
                'avg_final_hp_pct_p2': avg_hp_pct_p2,
                "player_won" : battle["player_won"]
            })
    
    return pd.DataFrame(final)

