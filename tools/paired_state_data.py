"""Generate state tasks with paired paraphrases and counterfactual outcomes."""

import random

from codes.sft_data import encode_sft_example
from .state_tasks import FILLER, PLACES, TEST_NAMES, TRAIN_NAMES, make_task, render_event, solve


CONTEXT = 256
TRAIN_EVENTS = (0, 1, 4, 5)
HELD_OUT_EVENTS = (2, 3)
QUESTIONS = {
    "owner": (
        "Who has the {item} now? Answer with one name.",
        "Which person is holding the {item} at the end? Give just the name.",
        "Whose hands is the {item} in now? Reply with a name.",
        "Name the person who ended up with the {item}.",
        "At the end of the story, who is carrying the {item}?",
    ),
    "location": (
        "Where is the {item} now? Answer with one place.",
        "At the end, where can the {item} be found? Give just the place.",
        "Where did the {item} end up? Reply with a place.",
        "Name the place that now holds the {item}.",
        "At the end of the story, where should someone look for the {item}?",
    ),
}


def event_text(event, domain, style):
    if style < 4 or event["kind"] == "swap":
        return render_event(event, domain, style)
    item, source, destination = (event[key] for key in ("object", "source", "destination"))
    if domain == "owner":
        return (
            f"The {item} passed from {source} to {destination}.",
            f"{destination} took the {item} that {source} offered.",
        )[style - 4]
    actor = event["actor"]
    return (
        f"The {item} left the {source} and ended up at the {destination} when {actor} moved it.",
        f"After lifting the {item} from the {source}, {actor} set it at the {destination}.",
    )[style - 4]


def chain_key(task):
    """Identify an event chain regardless of phrasing, question, or actor of a move."""
    events = tuple(
        (e["kind"], *sorted((e["a"], e["b"]))) if e["kind"] == "swap" else
        (e["kind"], e["object"], e["source"], e["destination"])
        for e in task["events"]
    )
    return task["domain"], tuple(sorted(task["initial"].items())), events


def make_group(rng, tokenizer, split, index):
    """Render two valid outcomes under the same two wording combinations."""
    combined = split == "combined"
    held_out = split in ("wording", "combined")
    task = make_task(rng, tokenizer, "four_hops" if combined else "train", index,
                     names=TEST_NAMES if combined else TRAIN_NAMES)
    domain, target, initial = (task[key] for key in ("domain", "target", "initial"))
    prefix = task["events"][:-2]
    sources = {item: solve(initial, prefix, item) for item in initial}
    candidates = task["names"] if domain == "owner" else PLACES
    destinations = rng.sample([value for value in candidates if value not in sources.values()], 2)
    final_order = list(initial)
    initial_order = list(initial)
    rng.shuffle(final_order)
    rng.shuffle(initial_order)
    actor = rng.choice(task["names"])
    initial_text = [f"{initial[item]} had the {item}." if domain == "owner" else
                    f"The {item} was at the {initial[item]}." for item in initial_order]
    filler = []
    distance = 90 if combined else task["distance"]
    filler_order = rng.sample(FILLER, len(FILLER))
    while len(tokenizer.encode(" ".join(filler)).ids) < distance:
        filler.append(filler_order[len(filler) % len(filler_order)])

    template_pool = HELD_OUT_EVENTS if held_out else TRAIN_EVENTS
    event_styles = [rng.sample(template_pool, 2) for _ in task["events"]]
    question_styles = rng.sample((1, 4) if held_out else (0, 2, 3), 2)
    group_id = f"{split}-{index:05d}"
    result = []
    for outcome, endpoints in enumerate((destinations, destinations[::-1])):
        final_events = [
            {"kind": "move", "object": item, "source": sources[item],
             "destination": destination, "actor": actor}
            for item, destination in zip(final_order, endpoints)
        ]
        events = prefix + final_events
        answer = solve(initial, events, target)
        for wording, question_style in enumerate(question_styles):
            sentences = initial_text + [event_text(event, domain, styles[wording])
                                        for event, styles in zip(events, event_styles)]
            question = QUESTIONS[domain][question_style].format(item=target)
            result.append({**task, "id": f"{group_id}-{outcome}-{wording}",
                           "group_id": group_id, "outcome": outcome, "wording": wording,
                           "event_styles": [styles[wording] for styles in event_styles],
                           "question_style": question_style, "category": split,
                           "events": events, "answer": answer, "response": " " + answer,
                           "distance": distance, "story": " ".join(sentences),
                           "prompt": " ".join([*sentences, *filler, question])})
    return result


def make_datasets(tokenizer, group_count, test_groups, seed):
    """Build balanced datasets and reserve whole semantic chains across splits."""
    used_chains = set()
    used_prompts = set()
    datasets = {}
    for offset, (split, count) in enumerate((
        ("paired", group_count), ("seen", test_groups), ("wording", test_groups),
        ("combined", test_groups), ("control", 4 * group_count),
    )):
        rng = random.Random(seed + offset)
        rows = []
        for _ in range(100 * count):
            if len(rows) == (count if split == "control" else 4 * count):
                break
            if split == "control":
                task = make_task(rng, tokenizer, "train", len(rows))
                group = [{**task, "id": f"control-{len(rows):05d}"}]
            else:
                group = make_group(rng, tokenizer, split, len(rows) // 4)
            chains = {chain_key(row) for row in group}
            prompts = {row["prompt"] for row in group}
            if used_chains & chains or used_prompts & prompts or len(prompts) != len(group):
                continue
            # Reserve room for the four-token free-answer evaluator and chat markers.
            if any(len(tokenizer.encode(row["prompt"]).ids) + 7 > CONTEXT for row in group):
                continue
            for row in group:
                encode_sft_example(tokenizer, row["prompt"], row["response"], CONTEXT)
            rows.extend(group)
            used_chains.update(chains)
            used_prompts.update(prompts)
        else:
            raise RuntimeError(f"Could not fill the {split} split with distinct chains.")
        datasets[split] = rows
    return datasets
