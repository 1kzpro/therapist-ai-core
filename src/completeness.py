from typing import List, Dict, Any
from utils import deep_get

# Minimal set; extend to your needs
REQUIRED_FIELDS = [
    "patient_id",
    "visit_date",
    "visit_time",
    "age",
    "sex",
    "complaints",  # expect non-empty array
    "skin_and_mucosa.temperature"
]

def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False

def missing_fields(case_state: Dict[str, Any]) -> List[str]:
    missing = []
    for dotted in REQUIRED_FIELDS:
        v = deep_get(case_state, dotted)
        if is_empty(v):
            missing.append(dotted)
    return missing
