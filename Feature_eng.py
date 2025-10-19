import json
import numpy as np
import pandas as pd
import os


def get_dict_from_json(file_path):
    l=[]
    with open(file_path, 'r') as f:
        for line in f: #ok
            # json.loads() parses one line (one JSON object) into a Python dictionary
            l.append(json.loads(line))
    return l


def pokedex(data: list[dict]) -> pd.DataFrame:
    """Create a simple DataFrame with basic features for each observed pokemon in p1 team and p2_lead in the input data"""

    pokedex = []
    for battle in data:
        features = {}

        p1_team = battle.get('p1_team_details', []) # If that key doesn’t exist, return an empty list []
        if p1_team:
            for pokemon in p1_team:
                features['name'] = pokemon.get('name')
                features['level'] = pokemon.get('level')
                features['type1'] = pokemon.get('types')[0]
                features['type2'] = pokemon.get('types')[1] 
                features['base_hp'] = pokemon.get('base_hp')
                features['base_atk'] = pokemon.get('base_atk')
                features['base_def'] = pokemon.get('base_def')
                features['base_spa'] = pokemon.get('base_spa')
                features['base_spd'] = pokemon.get('base_spd')
                features['base_spe'] = pokemon.get('base_spe')

        p2_lead = battle.get('p2_lead_details',[])
        if p2_lead:
            features['name'] = p2_lead.get('name')
            features['level'] = p2_lead.get('level')
            features['type1'] = p2_lead.get('types')[0]
            features['type2'] = p2_lead.get('types')[1] 
            features['base_hp'] = p2_lead.get('base_hp')
            features['base_atk'] = p2_lead.get('base_atk')
            features['base_def'] = p2_lead.get('base_def')
            features['base_spa'] = p2_lead.get('base_spa')
            features['base_spd'] = p2_lead.get('base_spd')
            features['base_spe'] = p2_lead.get('base_spe')

        
        pokedex.append(features)

    return pd.DataFrame(pokedex)


def opponents_pokemon(data: list[dict]) -> pd.DataFrame:
    """Create a simple DataFrame with basic features for each observed opponent's pokemon """
    pokedex = []
    for battle in data:
        features = {}
        battle_cox=battle.get("battle_timeline",[])
        if battle_cox:
            for turn in battle_cox:

                features['name']= turn.get("p2_pokemon_state").get('name')
                
                pokedex.append(features)

    return pd.DataFrame(pokedex)


def get_all_def_types(data: list[dict]) -> pd.DataFrame:
    """Create a DataFrame with all unique pokemon defense types observed in the dataset"""

    t1 = pokedex(data)['type1'].drop_duplicates().reset_index(drop=True)
    t2 = pokedex(data)['type2'].drop_duplicates().reset_index(drop=True)
    all_types = pd.concat([t1, t2]).dropna().drop_duplicates().reset_index(drop=True)
    
    return all_types


# Define the 19 types 
types = [
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
    "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
    "psychic", "rock", "steel", "stellar", "water"
]

# Define the dictionary for easy access
type_to_index = {t: i for i, t in enumerate(types)}

# Define the type effectiveness chart as a 2D numpy array.
# Rows = attacker, Columns = defender.
# Values: 0 = no effect, 0.5 = not very effective, 1 = normal, 2 = super effective.

# the matrix comes from: https://www.smogon.com/dex/sv/types/ depends on the pokemon generation....

type_chart = np.array([
#   bug  dar  dra  ele  fai  fig  fir  fly  gho  gra  gro  ice  nor  poi  psy  roc  ste  stl  wat
    [1,   2,   1,   1,   1,   0.5, 0.5, 0.5, 1,   2,   1,   1,   1,   0.5, 2,   1,   1,   1,   1],  # bug
    [1,   1,   1,   1,   0.5, 0.5, 1,   1,   2,   1,   1,   1,   1,   1,   2,   1,   1,   1,   1],  # dark
    [1,   1,   2,   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0.5, 1,   1],  # dragon
    [1,   1,   0.5, 0.5, 1,   1,   1,   2,   1,   0.5, 0,   1,   1,   1,   1,   1,   1,   1,   2],  # electric
    [1,   2,   1,   1,   1,   2,   0.5, 1,   1,   1,   1,   1,   1,   0.5, 1,   1,   0.5, 1,   1],  # fairy
    [2,   2,   1,   1,   0.5, 1,   1,   0.5, 0,   1,   1,   2,   2,   0.5, 0.5, 2,   2,   1,   1],  # fighting
    [2,   1,   0.5, 1,   1,   1,   0.5, 1,   1,   2,   1,   2,   1,   1,   1,   0.5, 2,   1,   0.5], # fire
    [2,   1,   1,   0.5, 1,   2,   1,   1,   1,   2,   0,   1,   1,   1,   1,   0.5, 0.5, 1,   1],  # flying
    [1,   0.5, 1,   1,   1,   1,   1,   1,   2,   1,   1,   1,   0,   1,   2,   1,   1,   1,   1],  # ghost
    [0.5, 1,   0.5, 1,   0.5, 0.5, 0.5, 0.5, 1,   0.5, 2,   1,   1,   0.5, 1,   2,   0.5, 1,   2],  # grass
    [1,   1,   1,   2,   1,   1,   2,   0,   1,   0.5, 1,   1,   1,   2,   1,   2,   1,   1,   1],  # ground
    [1,   1,   2,   1,   1,   1,   0.5, 2,   1,   1,   2,   0.5, 1,   1,   1,   1,   0.5, 1,   0.5], # ice
    [1,   1,   1,   1,   1,   1,   1,   1,   0,   1,   1,   1,   1,   1,   1,   0.5, 0.5, 1,   1],  # normal
    [1,   1,   1,   1,   0.5, 1,   1,   1,   0.5, 2,   0.5, 1,   1,   0.5, 1,   0.5, 0,   1,   1],  # poison
    [1,   0,   1,   1,   1,   2,   1,   1,   1,   1,   1,   1,   1,   2,   0.5, 1,   0.5, 1,   1],  # psychic
    [2,   1,   1,   1,   1,   0.5, 2,   2,   1,   1,   0.5, 2,   1,   1,   1,   0.5, 0.5, 1,   2],  # rock
    [1,   1,   1,   0.5, 2,   1,   0.5, 1,   1,   1,   1,   2,   1,   1,   1,   2,   0.5, 1,   0.5], # steel
    [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],  # stellar
    [1,   1,   0.5, 1,   1,   1,   2,   1,   1,   0.5, 1,   1,   1,   1,   1,   1,   1,   1,   0.5]  # water
])


def effectiveness(attacking_type: str, defending_types: list[str]) -> float:
    '''Calculate the effectiveness of an attack type against one or two defending types.'''

    idx_att = type_to_index[attacking_type.lower()]
    mult = 1.0
    for d in defending_types:
        d = d.lower()
        if d != 'notype':
            idx_def = type_to_index[d]
            mult *= type_chart[idx_att, idx_def]
    return mult


def get_dict_def_types(data: list[dict]) -> dict:
    """Create a dictionary with pokemon name as key and a set of its types as value"""
    pokemon_def_types = dict()
    pokemons = pokedex(data)[['name','type1','type2']].drop_duplicates().sort_values('name').reset_index(drop=True)

    # Create the dictionary
    pokemon_def_types = pokemons.set_index('name').apply(lambda row: {t for t in row if t != 'notype'}, axis=1).to_dict()

    return pokemon_def_types



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
                'avg_diff': avg_eff_p1-avg_eff_p2,
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
                'avg_diff': avg_eff_p1-avg_eff_p2,
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


def get_dict_base_stats(data: list[dict]) -> dict:
    """Create a dictionary with pokemon name as key and a list of its stats as value: base_atk	base_def base_spa base_spd"""

    pokemon_def_types = dict()
    pokemons = pokedex(data)[['name','base_atk','base_def','base_spa','base_spd']].drop_duplicates().sort_values('name').reset_index(drop=True)

    # Create the dictionary
    pokemon_def_types = pokemons.set_index('name').apply(lambda row: [t for t in row ], axis=1).to_dict()

    return pokemon_def_types


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
