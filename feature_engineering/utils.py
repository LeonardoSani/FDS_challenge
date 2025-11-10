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
    
    """Create a set with all unique status conditions observed in the dataset"""
    status_conditions = set()
    for battle in data:
        for turn in battle['battle_timeline']:
            p1_status = turn['p1_pokemon_state']['status']
            p2_status = turn['p2_pokemon_state']['status']
            status_conditions.add(p1_status)
            status_conditions.add(p2_status)
    return status_conditions


def get_all_effects(data: list[dict]) -> set:
    """Create a set with all unique effects observed in the dataset"""
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


def get_last_hp(battle: dict, include_fainted: bool = False) -> dict:
    """
    Finds the last observed HP percentage for each Pokémon that participated
    in the battle timeline. Assumes no null data.

    Args:
        battle: The battle dictionary (complete structure).
        include_fainted: If True, includes fainted Pokémon (hp_pct == 0)
                         in the returned dict with a value of 0.0.
                         If False (default), only includes Pokémon whose
                         last observed hp_pct > 0.

    Returns:
        A dictionary with keys 'observed_P1' and 'observed_P2'.
        Each key holds another dictionary mapping Pokémon names (str) to
        their last observed HP percentage (float).
    """
    p1_last_hp = {}
    p2_last_hp = {}

    # Iterate through the timeline to find the last-seen HP
    for turn in battle["battle_timeline"]:
        # Direct access, assuming these keys always exist
        p1_state = turn["p1_pokemon_state"]
        p2_state = turn["p2_pokemon_state"]
        
        p1_last_hp[p1_state["name"]] = p1_state["hp_pct"]
        p2_last_hp[p2_state["name"]] = p2_state["hp_pct"]

    # Filter the results based on the include_fainted flag
    result = {"observed_P1": {}, "observed_P2": {}}

    for name, hp in p1_last_hp.items():
        if hp > 0:
            result["observed_P1"][name] = hp
        elif include_fainted:
            result["observed_P1"][name] = 0.0

    for name, hp in p2_last_hp.items():
        if hp > 0:
            result["observed_P2"][name] = hp
        elif include_fainted:
            result["observed_P2"][name] = 0.0

    return result


def get_p1_bench(battle: dict) -> set:
    """
    Finds the set of Pokémon on P1's team that never appeared in the
    battle timeline. Assumes no null data.

    Args:
        battle: The battle dictionary (complete structure).

    Returns:
        A set of Pokémon names (str) that were in 'p1_team_details'
        but not in the 'battle_timeline'.
    """
    
    
    full_team_names = {p["name"] for p in battle["p1_team_details"]}

    
    appeared_names = {
        turn["p1_pokemon_state"]["name"]
        for turn in battle["battle_timeline"]
    }

    bench_names = full_team_names - appeared_names

    return bench_names


def team_potential(A: list[str], B: list[str], type_dict: dict) -> dict:
    """
    (ML-Ready Robust Version)
    Compute type-based potential of team A (attacker) vs team B (defender).
    
    This version explicitly handles the 3 scenarios for a ML model:
    1. A is empty (0 potential)     -> returns all 0.0
    2. A not empty, B empty (max advantage) -> returns max values
    3. A not empty, B not empty (standard case) -> runs calculation
    """

    POTENTIAL_METRIC_KEYS = [
    'avg_best_potential', 'min_best_potential', 'max_best_potential',
    'avg_redundancy', 'min_redundancy', 'coverage_fraction', 'entropy'
    ]

    # This is the "floor" - we have no attackers, so we have 0 potential.
    DEFAULT_METRICS_NO_POTENTIAL = {key: 0.0 for key in POTENTIAL_METRIC_KEYS}

    # --- Case 1: Attacker set A is empty ---
    # We have no attackers, so we have 0 potential. This is the worst case.
    if not A:
        return DEFAULT_METRICS_NO_POTENTIAL

    # --- Case 2: Attacker set A is NOT empty, but Defender set B is empty ---
    # We have an advantage (we have members, they don't).
    # This is the best case. We return a set of "max" values.
    if not B:
        # We've already won.
        # We set potential to 4.0 (max effectiveness).
        # We set coverage to 1.0 (100%).
        # We set redundancy to the number of attackers we have.
        # Entropy is 0.0 as there's no "spread" against 0 defenders.
        return {
            'avg_best_potential': 4.0,
            'min_best_potential': 4.0,
            'max_best_potential': 4.0,
            'avg_redundancy': float(len(A)),
            'min_redundancy': float(len(A)),
            'coverage_fraction': 1.0,
            'entropy': 0.0
        }

    # --- Case 3: Both A and B have members ---
    # This is the standard "in-progress" battle.
    # We run your full calculation (with safety checks).
    else:
        best_vs = []
        redundancy = []
        
        for b_name in B:
            # Safety check for dictionary lookup
            def_types = list(type_dict.get(b_name, ['notype']))
            
            eff_list = []
            for a_name in A:
                # Safety check for dictionary lookup
                att_types = list(type_dict.get(a_name, ['notype']))
                
                try:
                    # We assume effectiveness() exists and works
                    eff = max(effectiveness(att_type, def_types) for att_type in att_types)
                except Exception:
                    eff = 1.0 # Default fallback
                eff_list.append(eff)
            
            # This is safe, since 'A' is not empty
            best_val = max(eff_list)
            strong_count = sum(e >= 2.0 for e in eff_list)
            best_vs.append(best_val)
            redundancy.append(strong_count)
        
        # This is safe, since 'B' is not empty
        best_vs_np = np.array(best_vs, dtype=float)
        redundancy_np = np.array(redundancy, dtype=float)

        # --- Aggregates ---
        avg_best = float(np.mean(best_vs_np))
        min_best = float(np.min(best_vs_np))
        max_best = float(np.max(best_vs_np))
        avg_redund = float(np.mean(redundancy_np))
        min_redund = float(np.min(redundancy_np))
        coverage = float(np.mean(best_vs_np >= 2.0))

        # --- Entropy ---
        if best_vs_np.sum() > 0:
            p = best_vs_np / best_vs_np.sum()
            entropy = -float(np.sum(p * np.log(p + 1e-12)))
            entropy_divisor = np.log(len(B)) if len(B) > 1 else 1.0
            if entropy_divisor > 0:
                entropy /= entropy_divisor
        else:
            entropy = 0.0

        # Return the dictionary of calculated metrics
        return {
            'avg_best_potential': avg_best,
            'min_best_potential': min_best,
            'max_best_potential': max_best,
            'avg_redundancy': avg_redund,
            'min_redundancy': min_redund,
            'coverage_fraction': coverage,
            'entropy': entropy
        }
