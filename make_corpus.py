"""Generate the Experiment B teaching corpus: negation and spatial relations.

Writes corpus/negation.txt and corpus/spatial.txt, one short story per line.
All text is original synthetic material written for this assignment.

Separation rules (the eval suite is the exam; this is study material):
  * The exact eval stories, answer lists and outputs never appear.
  * An eval's subject never appears in the pattern it tests: "box"/"door" are never
    the subject of a property correction, and "ava" never has an action correction.
  * The eval answer pairs are never taught as a correction (red->blue, tea->milk,
    open->closed), so a correct eval answer must come from copying the context,
    not from a memorized pair.
  * No story contains 3+ content words from any single eval case (see check_leakage.py).
  * Names used by the reference-category evals are not used, to keep untargeted
    categories out of this experiment.

Why sentences are joined with ".word" (no space after the period): the notebook's
chunk_text() splits passages at "[.!?] + whitespace", so "a . b ." would become two
unrelated one-sentence passages and the model could never learn a pattern that spans
sentences. "a .b ." tokenizes to exactly the same tokens ("a", ".", "b", ".") while
staying one passage. Pass --split-sentences to write the ordinary " . " spacing
instead (used for the Experiment C ablation).
"""

import argparse
import random
from pathlib import Path

SEED = 7
TARGET_PER_FILE = 1600

# ---- shared pools -----------------------------------------------------------
NAMES = [
    ("mia", "she"),
    ("zoe", "she"),
    ("lily", "she"),
    ("ruby", "she"),
    ("iris", "she"),
    ("lucy", "she"),
    ("anna", "she"),
    ("rosa", "she"),
    ("jade", "she"),
    ("yuki", "she"),
    ("sam", "he"),
    ("max", "he"),
    ("ben", "he"),
    ("kai", "he"),
    ("jack", "he"),
    ("owen", "he"),
    ("theo", "he"),
    ("ivan", "he"),
    ("hugo", "he"),
    ("ravi", "he"),
]

# ---- negation ---------------------------------------------------------------
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "pink",
    "gray",
    "purple",
]
COLOR_OBJECTS = [
    "cup",
    "hat",
    "chair",
    "shirt",
    "pen",
    "kite",
    "mug",
    "sock",
    "scarf",
    "coat",
    "bowl",
    "towel",
    "flag",
    "rug",
    "vase",
    "bucket",
    "bottle",
    "jar",
    "car",
    "bus",
    "ball",
    "book",
    "lamp",
    "bag",
    "plate",
    "clock",
    "sofa",
    "basket",
]
STATES = [
    "open",
    "closed",
    "locked",
    "wide",
    "missing",
    "broken",
    "clean",
    "dirty",
    "old",
    "ready",
]
STATE_OBJECTS = [
    "window",
    "gate",
    "drawer",
    "cabinet",
    "lid",
    "fridge",
    "oven",
    "garage",
    "closet",
    "suitcase",
    "shop",
    "cage",
]
SOLIDS = ["rice", "bread", "soup", "cake", "cheese", "eggs", "pasta", "salad", "apple",
          "banana", "pear", "peach", "mango", "honey", "beans", "noodles"]
DRINKS = ["tea", "milk", "juice", "coffee", "lemonade", "cocoa", "cider"]
COOKED = ["rice", "soup", "pasta", "eggs", "beans", "noodles"]
FOODS = SOLIDS + DRINKS
THINGS = [
    "pen",
    "cup",
    "hat",
    "ticket",
    "map",
    "key",
    "phone",
    "coat",
    "book",
    "bag",
    "ball",
    "lamp",
    "clock",
    "towel",
    "scarf",
    "basket",
]
# (base, past, item pool)
VERBS = [
    ("buy", "bought", FOODS + THINGS),
    ("order", "ordered", FOODS),
    ("eat", "ate", SOLIDS),
    ("drink", "drank", DRINKS),
    ("cook", "cooked", COOKED),
    ("bring", "brought", FOODS + THINGS),
    ("take", "took", THINGS),
    ("find", "found", THINGS),
    ("grab", "grabbed", THINGS),
    ("carry", "carried", THINGS),
    ("choose", "chose", FOODS + THINGS),
]

# Never taught as a correction, in either order (these are eval answer pairs).
BANNED_PAIRS = [{"red", "blue"}, {"tea", "milk"}, {"open", "closed"}]


def property_story(rng):
    if rng.random() < 0.5:
        obj, values = rng.choice(COLOR_OBJECTS), COLORS
    else:
        obj, values = rng.choice(STATE_OBJECTS), STATES
    a, b, c = rng.sample(values, 3)
    if {a, b} in BANNED_PAIRS:
        return None
    name = rng.choice(NAMES)[0]
    frames = [
        "the {o} is not {a} .it is {b} .the {o} is {b} .",
        "the {o} was not {a} .it was {b} .the {o} was {b} .",
        "{n} said the {o} is {a} .that is wrong .the {o} is not {a} .it is {b} .",
        "is the {o} {a} ?no .the {o} is not {a} .the {o} is {b} .",
        "the {o} is {b} , not {a} .so the {o} is {b} .",
        "{n} thought the {o} was {a} .it was not {a} .it was {b} .",
        "the {o} is not {a} and it is not {c} .it is {b} .the {o} is {b} .",
        "we looked at the {o} .it is not {a} .it is {b} .yes , the {o} is {b} .",
    ]
    if {b, c} in BANNED_PAIRS or {a, c} in BANNED_PAIRS:
        frames = frames[:6] + frames[7:]
    return rng.choice(frames).format(o=obj, a=a, b=b, c=c, n=name)


def action_story(rng):
    name, pronoun = rng.choice(NAMES)
    base, past, pool = rng.choice(VERBS)
    x, y = rng.sample(pool, 2)
    if {x, y} in BANNED_PAIRS:
        return None
    article = lambda item: item if item in FOODS else "the " + item
    frames = [
        "{n} did not {v} {x} .{p} {vd} {y} .{n} {vd} {y} .",
        "{n} did not {v} {x} .{p} {vd} {y} instead .so {n} {vd} {y} .",
        "{n} wanted to {v} {x} , but {p} did not .{p} {vd} {y} .{n} {vd} {y} .",
        "it was not {x} that {n} {vd} .{n} {vd} {y} .",
        "{n} {vd} {y} , not {x} .{n} {vd} {y} .",
        "{n} did not {v} {x} today .{p} {vd} {y} .what did {n} {v} ?{n} {vd} {y} .",
    ]
    return rng.choice(frames).format(n=name, p=pronoun, v=base, vd=past, x=article(x), y=article(y))


# ---- spatial relations ------------------------------------------------------
ITEMS = [
    "lamp",
    "desk",
    "shelf",
    "book",
    "bag",
    "ball",
    "box",
    "cup",
    "plate",
    "chair",
    "table",
    "bed",
    "rug",
    "clock",
    "pen",
    "key",
    "phone",
    "mirror",
    "window",
    "door",
    "sofa",
    "basket",
    "bottle",
    "jar",
    "hat",
    "coat",
    "bowl",
    "vase",
    "picture",
    "sink",
]
CONTAINERS = [
    "bag",
    "box",
    "drawer",
    "jar",
    "basket",
    "cabinet",
    "bowl",
    "fridge",
    "suitcase",
    "closet",
    "cup",
    "bottle",
]
SMALL = [
    "book",
    "pen",
    "key",
    "phone",
    "ball",
    "cup",
    "coin",
    "letter",
    "ring",
    "toy",
    "card",
    "sock",
    "map",
    "ticket",
    "apple",
    "banana",
    "pear",
    "peach",
    "mango",
]
PLACES = [
    "farm",
    "lake",
    "park",
    "river",
    "bridge",
    "village",
    "tower",
    "field",
    "forest",
    "beach",
    "store",
    "bank",
    "school",
    "hospital",
    "station",
    "market",
    "office",
]
INVERSE = {
    "above": "below",
    "below": "above",
    "left": "right",
    "right": "left",
    "north": "south",
    "south": "north",
    "on": "under",
    "under": "on",
}
PEOPLE = [n for n, _ in NAMES] + ["ava"]


def vertical_story(rng):
    x, y = rng.sample(ITEMS, 2)
    r = rng.choice(["above", "below"])
    inv = INVERSE[r]
    name = rng.choice(PEOPLE)
    frames = [
        "the {x} is {r} the {y} .the {y} is {i} the {x} .",
        "the {x} is {r} the {y} .so the {y} is {i} the {x} .",
        "{n} put the {x} {r} the {y} .now the {y} is {i} the {x} .",
        "the {x} is {r} the {y} .where is the {y} ?the {y} is {i} the {x} .",
        "the {y} is {i} the {x} .the {x} is {r} the {y} .",
    ]
    return rng.choice(frames).format(x=x, y=y, r=r, i=inv, n=name)


def horizontal_story(rng):
    if rng.random() < 0.75:
        x, y = rng.sample(ITEMS, 2)
    else:
        x, y = rng.sample(PEOPLE, 2)
    r = rng.choice(["left", "right"])
    inv = INVERSE[r]
    art = "" if x in PEOPLE else "the "
    frames = [
        "{a}{x} is {r} of {a}{y} .{a}{y} is to the {i} of {a}{x} .",
        "{a}{x} is to the {r} of {a}{y} .{a}{y} is to the {i} of {a}{x} .",
        "{a}{x} is {r} of {a}{y} .so {a}{y} is to the {i} .",
        "{a}{x} is {r} of {a}{y} .where is {a}{y} ?{a}{y} is to the {i} of {a}{x} .",
        "{a}{y} is to the {i} of {a}{x} .{a}{x} is to the {r} of {a}{y} .",
    ]
    return rng.choice(frames).format(a=art, x=x, y=y, r=r, i=inv)


def compass_story(rng):
    x, y = rng.sample(PLACES, 2)
    r = rng.choice(["north", "south"])
    inv = INVERSE[r]
    frames = [
        "the {x} is {r} of the {y} .the {y} is {i} of the {x} .",
        "the {x} is {r} of the {y} .so the {y} is to the {i} .",
        "we drove from the {y} to the {x} .the {x} is {r} of the {y} .the {y} is {i} of the {x} .",
    ]
    return rng.choice(frames).format(x=x, y=y, r=r, i=inv)


def container_story(rng):
    c = rng.choice(CONTAINERS)
    x = rng.choice([s for s in SMALL if s != c])
    name = rng.choice(PEOPLE)
    frames = [
        "the {x} is inside the {c} .the {c} contains the {x} .",
        "the {c} contains the {x} .the {x} is inside the {c} .",
        "{n} put the {x} inside the {c} .now the {c} contains the {x} .",
        "the {x} is inside the {c} .what does the {c} contain ?the {c} contains the {x} .",
        "the {x} is not on the table .it is inside the {c} .the {c} contains the {x} .",
    ]
    return rng.choice(frames).format(x=x, c=c, n=name)


def on_under_story(rng):
    x, y = rng.sample(ITEMS, 2)
    r = rng.choice(["on", "under"])
    frames = [
        "the {x} is {r} the {y} .the {y} is {i} the {x} .",
        "the {x} is beside the {y} .the {y} is beside the {x} .",
        "{n} sat beside the {y} .the {y} is beside {n} .",
    ]
    name = rng.choice(PEOPLE)
    return rng.choice(frames).format(x=x, y=y, r=r, i=INVERSE[r], n=name)


def build(generators, rng, target, violates):
    stories, rejected = set(), 0
    while len(stories) < target:
        generator = rng.choices(
            [g for g, _ in generators], weights=[w for _, w in generators]
        )[0]
        story = generator(rng)
        if story is None or violates(story):
            rejected += 1
            continue
        stories.add(story)
    return sorted(stories), rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="corpus")
    parser.add_argument(
        "--split-sentences",
        action="store_true",
        help='write ordinary " . " spacing, so the notebook splits every sentence',
    )
    args = parser.parse_args()
    from check_leakage import load_cases, story_violations

    cases = load_cases()
    violates = lambda story: bool(story_violations(story, cases))
    rng = random.Random(SEED)
    negation, rej_n = build(
        [(property_story, 1), (action_story, 1)], rng, TARGET_PER_FILE, violates
    )
    spatial, rej_s = build(
        [
            (vertical_story, 3),
            (horizontal_story, 3),
            (compass_story, 1),
            (container_story, 3),
            (on_under_story, 1),
        ],
        rng,
        TARGET_PER_FILE,
        violates,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, stories in [("negation.txt", negation), ("spatial.txt", spatial)]:
        rng.shuffle(stories)
        text = "\n".join(stories) + "\n"
        if args.split_sentences:
            text = (
                text.replace(" .", " . ")
                .replace(" ?", " ? ")
                .replace(" \n", "\n")
                .replace("  ", " ")
            )
        (out / filename).write_text(text, encoding="utf-8")
    print(
        f"negation: {len(negation)} stories ({rej_n} candidates rejected by separation rules)"
    )
    print(
        f"spatial:  {len(spatial)} stories ({rej_s} candidates rejected by separation rules)"
    )


if __name__ == "__main__":
    main()
