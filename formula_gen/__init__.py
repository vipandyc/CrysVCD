import pathlib
import os

PARENT_PATH = pathlib.Path(__file__).parent.resolve()
IONIC_COMPOSITIONS_PATH = os.path.join(
    PARENT_PATH,
    'data',
    'Chemical_formulas_with_valence_ionic.pkl'
)
ALLOY_COMPOSITIONS_PATH = os.path.join(
    PARENT_PATH,
    'data',
    'Chemical_formulas_with_valence_alloy.pkl'
)