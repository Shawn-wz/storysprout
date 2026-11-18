"""Generate and verify short ownership and location stories from explicit states."""

import random
import re

TRAIN_NAMES = ("Mia", "Ben", "Tom", "Lily", "Sam", "Max", "Tim", "Anna")
TEST_NAMES = ("Lucy", "Jack", "Bob", "Jane", "Emma", "Kate", "Amy", "Leo")
OBJECTS = ("key", "book", "ball", "hat", "coin", "doll")
PLACES = ("box", "bag", "desk", "shelf", "basket", "drawer")
FILLER = (
    "A bird sang in a tree. The sun was warm and the sky was blue.",
    "A yellow butterfly flew over the flowers near the garden gate.",
    "The wind moved the leaves and a small white cloud passed overhead.",
    "A rabbit sat by the path and watched the bright flowers.",
    "Beyond the garden there was a quiet road and a little white house.",
    "A tiny bell rang in the distance and then everything was quiet again.",
)


def solve(initial: dict, events: list[dict], target: str) -> str:
    """Replay state changes independently of any language-model output."""
    state = dict(initial)
    for event in events:
        if event["kind"] == "swap":
            a, b = event["a"], event["b"]
            state = {obj: b if owner == a else a if owner == b else owner
                     for obj, owner in state.items()}
        else:
            if state[event["object"]] != event["source"]:
                raise ValueError("Event source contradicts the current state.")
            state[event["object"]] = event["destination"]
    return state[target]


def render_event(event: dict, domain: str, style: int) -> str:
    """Render equivalent state changes with balanced word orders."""
    if event["kind"] == "swap":
        return f"{event['a']} and {event['b']} swapped everything they were holding."
    obj, a, b = event["object"], event["source"], event["destination"]
    if domain == "owner":
        return (
            f"{a} gave the {obj} to {b}.",
            f"{b} got the {obj} from {a}.",
            f"{a} handed the {obj} over to {b}.",
            f"{b} received the {obj} from {a}.",
        )[style]
    actor = event["actor"]
    return (
        f"{actor} took the {obj} from the {a} and put it in the {b}.",
        f"The {obj} was moved from the {a} to the {b} by {actor}.",
        f"{actor} placed the {obj} on the {b} after taking it from the {a}.",
        f"After leaving the {a}, the {obj} ended up on the {b}.",
    )[style]


def make_task(rng: random.Random, tokenizer, category: str, index: int) -> dict:
    """Create a fully labeled task with independent distractor-object events."""
    domain = "owner" if index % 2 == 0 else "location"
    names = TEST_NAMES if category in ("names", "combined", "hard_ood") else TRAIN_NAMES
    style_pool = (2, 3) if category in ("paraphrase", "combined", "hard_ood") else (0, 1)
    if category.startswith("hard"):
        hops, distance = rng.choice((3, 4, 5)), rng.choice((90, 110, 130))
    elif category == "single":
        hops, distance = 1, 0
    elif category == "long":
        hops, distance = rng.choice((1, 2, 3)), 100
    elif category == "four_hops":
        hops, distance = 4, rng.choice((0, 40))
    elif category == "combined":
        hops, distance = 3, 100
    elif category in ("train", "dev", "rl"):
        hops, distance = rng.choice((1, 2, 3)), rng.choice((0, 30, 70))
    else:
        hops, distance = rng.choice((2, 3)), rng.choice((0, 40))
    people = rng.sample(list(names), 4)
    objects = rng.sample(list(OBJECTS), 2)
    target, other = objects
    values = people if domain == "owner" else list(PLACES)
    starting = rng.sample(values, 2)
    initial = dict(zip(objects, starting))
    narration = list(objects)
    rng.shuffle(narration)
    if domain == "owner":
        sentences = [f"{initial[obj]} had the {obj}." for obj in narration]
    else:
        sentences = [f"The {obj} was in the {initial[obj]}." for obj in narration]
    state = dict(initial)
    events = []
    for _ in range(hops):
        # Each target event is followed or preceded by a distinct-object distractor.
        order = [target, other]
        rng.shuffle(order)
        for obj in order:
            if domain == "owner" and obj == target and rng.random() < 0.15:
                a = state[target]
                b = rng.choice([p for p in people if p != a])
                event = {"kind": "swap", "a": a, "b": b}
            else:
                destination = rng.choice([v for v in values if v != state[obj]])
                event = {"kind": "move", "object": obj, "source": state[obj],
                         "destination": destination, "actor": rng.choice(people)}
            events.append(event)
            state = {obj: solve(initial, events, obj) for obj in objects}
            sentences.append(render_event(event, domain, rng.choice(style_pool)))
    filler = []
    while len(tokenizer.encode(" ".join(filler)).ids) < distance:
        filler.append(FILLER[len(filler) % len(FILLER)])
    if domain == "owner":
        question = (f"Who has the {target} now? Answer with one name."
                    if category not in ("paraphrase", "combined", "hard_ood")
                    else f"Which person is holding the {target} at the end? Give just the name.")
    else:
        question = (f"Where is the {target} now? Answer with one place."
                    if category not in ("paraphrase", "combined", "hard_ood")
                    else f"At the end, where can the {target} be found? Give just the place.")
    story = " ".join(sentences)
    prompt = " ".join([story, *filler, question])
    answer = solve(initial, events, target)
    return {
        "prompt": prompt, "response": " " + answer, "answer": answer,
        "domain": domain, "category": category, "hops": hops,
        "distance": distance, "names": list(names), "initial": initial,
        "events": events, "target": target, "story": story,
    }


def make_split(tokenizer, category: str, count: int, seed: int, seen: set) -> list[dict]:
    """Draw unique prompts that fit the unchanged 256-token model window."""
    rng = random.Random(seed)
    rows = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("Unable to fill the requested unique task split.")
        row = make_task(rng, tokenizer, category, len(rows))
        encoded_length = len(tokenizer.encode(row["prompt"]).ids)
        if encoded_length > 244 or row["prompt"] in seen:
            continue
        seen.add(row["prompt"])
        row["id"] = f"{category}-{len(rows):05d}"
        rows.append(row)
    return rows


def reward_ids(tokenizer, response: list[int], answer: str, special: dict) -> float:
    """Reward only a single correct answer, rejecting special-token insertions."""
    ids = list(response)
    if ids and ids[-1] == special["<eos>"]:
        ids.pop()
    if not ids or any(token in set(special.values()) for token in ids):
        return 0.0
    text = tokenizer.decode(ids, skip_special_tokens=False).strip()
    match = re.fullmatch(r"([A-Za-z]+)[.!]?", text)
    return float(bool(match) and match.group(1).casefold() == answer.casefold())
