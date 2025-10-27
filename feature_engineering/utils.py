# feature_engineering/utils.py

import json
import numpy as np
import pandas as pd
from .constants import type_chart, type_to_index # Import constants

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
    pokemon_def_types = pokemons.set_index('name').apply(lambda row: {t.lower() for t in row if t != 'notype'}, axis=1).to_dict()

    return pokemon_def_types


def get_dict_attacker_types(data: list[dict]) -> dict: 
    """Create a dictionary with pokemon name as key and a list of its types as value, this function do the same 
    as get_dict_def_types the only difference is we use a list instead of a set and the names is different to clarify the use case"""

    # Use pokedex to get the types of the attacking Pokémon (which are on the field)
    pokemons = pokedex(data)[['name','type1','type2']].drop_duplicates().sort_values('name').reset_index(drop=True)
    
    # Create the dictionary: name -> [type1, type2]
    # We use a list instead of a set/tuple here because we need to check both types for STAB.
    pokemon_att_types = pokemons.set_index('name').apply(lambda row: [t.lower() for t in row if t and t.lower() != 'notype'], axis=1).to_dict()

    return pokemon_att_types


def get_dict_base_stats(data: list[dict]) -> dict: 
    """Create a dictionary with pokemon name as key and a list of its stats as value: base_atk	base_def base_spa base_spd"""

    pokemon_stats = dict()
    pokemons = pokedex(data)[['name','base_atk','base_def','base_spa','base_spd']].drop_duplicates().sort_values('name').reset_index(drop=True)

    # Create the dictionary
    pokemon_stats = pokemons.set_index('name').apply(lambda row: [t for t in row ], axis=1).to_dict()

    return pokemon_stats


def get_dict_base_stats1(data: list[dict]) -> dict: 
    """Return a dict mapping pokemon name -> dict of base stats (base_hp, base_atk, base_def, base_spa, base_spd, base_spe)."""
    pokemons = pokedex(data)[['name','base_hp','base_atk','base_def','base_spa','base_spd','base_spe']].drop_duplicates().sort_values('name').reset_index(drop=True)

    # Create nested dict: name -> { stat_name: value, ... }
    stats_dict = pokemons.set_index('name').to_dict(orient='index')


    pokemon_stats = {
        name: {k: (int(v) if pd.notna(v) and float(v).is_integer() else (None if pd.isna(v) else float(v)))
               for k, v in stats.items()}
        for name, stats in stats_dict.items()
    }

    return pokemon_stats


def get_all_status_conditions(data: list[dict]) -> set:
    status_conditions = set()
    for battle in data:
        for turn in battle['battle_timeline']:
            p1_status = turn['p1_pokemon_state']['status']
            p2_status = turn['p2_pokemon_state']['status']
            status_conditions.add(p1_status)
            status_conditions.add(p2_status)
    return status_conditions


def get_all_effects(data: list[dict]) -> set:
    effects = set()
    for battle in data:
        for turn in battle['battle_timeline']:
            p1_effects = turn['p1_pokemon_state']['effects']
            p2_effects = turn['p2_pokemon_state']['effects']
            for effect in p1_effects:
                effects.add(effect)
            for effect in p2_effects:
                effects.add(effect)
    return effects