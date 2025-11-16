"""
Pokemon type effectiveness constants and charts.

This module contains the complete type system definitions and effectiveness
charts for Pokemon battle calculations.
"""
import numpy as np

# List of all Pokemon types in order
types = [
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
    "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
    "psychic", "rock", "steel", "stellar", "water"
]

# Mapping from type name to index for quick lookups
type_to_index = {t: i for i, t in enumerate(types)}

# Type effectiveness chart as a 2D numpy array
# Rows represent attacking types, columns represent defending types
# Values: 0 = no effect, 0.5 = not very effective, 1 = normal damage, 2 = super effective
# Source: https://www.smogon.com/dex/sv/types/

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
    [1,   1,   2,   1,   1,   1,   1,   2,   1,   1,   2,   0.5, 1,   1,   1,   1,   0.5, 1,   0.5], # ice
    [1,   1,   1,   1,   1,   1,   1,   1,   0,   1,   1,   1,   1,   1,   1,   0.5, 0.5, 1,   1],  # normal
    [1,   1,   1,   1,   0.5, 1,   1,   1,   0.5, 2,   0.5, 1,   1,   0.5, 1,   0.5, 0,   1,   1],  # poison
    [1,   0,   1,   1,   1,   2,   1,   1,   1,   1,   1,   1,   1,   2,   0.5, 1,   0.5, 1,   1],  # psychic
    [2,   1,   1,   1,   1,   0.5, 2,   2,   1,   1,   0.5, 2,   1,   1,   1,   0.5, 0.5, 1,   2],  # rock
    [1,   1,   1,   0.5, 2,   1,   0.5, 1,   1,   1,   1,   2,   1,   1,   1,   2,   0.5, 1,   0.5], # steel
    [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],  # stellar
    [1,   1,   0.5, 1,   1,   1,   2,   1,   1,   0.5, 1,   1,   1,   1,   1,   1,   1,   1,   0.5]  # water
])