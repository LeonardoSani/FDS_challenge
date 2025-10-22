import numpy as np

# Define the types 
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