# feature_engineering/__init__.py

from .extractors import (
    avg_effectiveness_1,
    avg_effectiveness_1_1,
    avg_effectiveness2,
    category_impact_score,
    avg_stab_multiplier,
    avg_final_HP_pct,
    avg_boost_diff_per_turn,
    avg_stat_diff_per_turn,
    accuracy_avg,
    granular_turn_counts,
    avg_team_vs_lead_stats,
    faint_count_diff_extractor,
    ratio_category_diff,
    calculate_voluntary_swap_diff,
    team_hp_advantage_flip_count,
    damage_efficiency_ratio,
    pokemon_encoding,
    avg_approx_damage,
    first_KO_momentum_feature,
)

from .utils import (
    get_dict_from_json,
    pokedex,            
    opponents_pokemon,  
    get_all_def_types,  
    effectiveness,      
    get_dict_def_types, 
    get_dict_attacker_types, 
    get_dict_base_stats,
    get_dict_base_stats1,
    get_all_status_conditions,
    get_all_effects,
    get_last_hp,
    get_p1_bench,
    team_potential
)
from .Aggregator import generate_features

__all__ = [
    # Extractor Functions
    'avg_effectiveness_1',
    'avg_effectiveness_1_1',
    'avg_effectiveness2',
    'category_impact_score',
    'avg_stab_multiplier',
    'avg_final_HP_pct',
    'avg_boost_diff_per_turn',
    'avg_stat_diff_per_turn',
    'accuracy_avg',
    'granular_turn_counts',
    'avg_team_vs_lead_stats',
    'faint_count_diff_extractor',
    'ratio_category_diff',
    'calculate_voluntary_swap_diff',
    'team_hp_advantage_flip_count',
    'damage_efficiency_ratio',
    'pokemon_encoding',
    'avg_approx_damage',
    'first_KO_momentum_feature',
    

    # Utility Functions
    'get_dict_from_json',
    'pokedex',
    'opponents_pokemon',
    'get_all_def_types',
    'effectiveness',
    'get_dict_def_types',
    'get_dict_attacker_types',
    'get_dict_base_stats',
    'get_dict_base_stats1',
    'get_all_status_conditions',
    'get_all_effects',
    'get_last_hp',
    'get_p1_bench',
    'team_potential',
    
    # Aggregator Function
    'generate_features'
]