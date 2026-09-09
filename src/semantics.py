CLASSES = [
    "crate",
    "barrel",
    "traffic cone",
    "tree trunk",
    "tree",
    "bench",
    "car",
    "brick wall",
    "fence",
    "street lamp",
    "planter",
]

ALIASES = {
    "lamp post": "street lamp",
    "metal pole": "street lamp",
}

PROMPT = ". ".join(CLASSES + list(ALIASES)) + "."

CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}

LOOKUP = dict(CLASS_INDEX)
LOOKUP.update(
    {alias: CLASS_INDEX[canonical] for alias, canonical in ALIASES.items()}
)

UNKNOWN = -1
FREE_COST = 1.0
UNKNOWN_COST = 5.0
OBSTACLE_COST = 100.0

DEFAULT_COSTS = {index: OBSTACLE_COST for index in range(len(CLASSES))}
DEFAULT_COSTS[UNKNOWN] = UNKNOWN_COST


def label_to_class_id(text):
    normalised = " ".join(str(text).lower().split())

    if normalised in LOOKUP:
        return LOOKUP[normalised]
 
    words = set(normalised.split()) #input text is split into words and stored in a set for comparison
    if not words:
        return UNKNOWN

    best_id = UNKNOWN
    best_key = (0, 0)

    for name, index in LOOKUP.items():
        parts = name.split()
        overlap = len(words & set(parts))
        if overlap == 0:
            continue
        key = (overlap, -len(parts))
        if key > best_key:
            best_key = key
            best_id = index

    return best_id
