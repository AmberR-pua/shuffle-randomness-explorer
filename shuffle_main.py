# Import necessary library
import math
import random
import time
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
# prepare for saving the metric result for different shuffle ways
CSV_DIR = Path("csv_file")
CSV_DIR.mkdir(exist_ok=True)

# This represents the deck as integers 0,1,2,...,n-1.
# our default card deck is size 52, as 13 cards of each of the four suits, ignoring the jokers.
def standard_deck(n: int = 52) -> List[int]:
    return list(range(n))

# collect everything needed for one shuffle way for simulation,
# contains name, parameter, whether to cheat, etc.
@dataclass(frozen=True)
class ShuffleConfig:
    name: str
    method: str
    params: Dict[str, object]
    shuffle_fn: Callable[[List[int], random.Random], None]
    cheat_mode: str = "none"
    cheat_cards: int = 0
    tracked_card: int = 0

### Shuffle Methods
# The implementations of the three primary shuffle ways we have
# for Riffle shuflle we use the gsr(Gilbert–Shannon–Reeds) model comes from the essay
def riffle_shuffle_gsr(deck: List[int], rng: random.Random, cut_p: float = 0.5) -> None:
    """
    This method implement a riffle shuffle way for shuffle cards, it change the card's order
    :param cut_p: conrol the probability of one card goes to left
    """
    if not (0.0 < cut_p < 1.0):
        raise ValueError("cut_p must be in (0,1)")

    n = len(deck)
    # simulate a binomial cut: for each of the n cards, decide whether it goes into the left packet independently
    cut = sum(1 for _ in range(n) if rng.random() < cut_p)

    left = deck[:cut]
    right = deck[cut:]

    out: List[int] = []
    i = j = 0
    # Continue until both packets are empty.
    while i < len(left) or j < len(right):
        # If left packet is empty, take only from right.
        if i == len(left):
            out.append(right[j])
            j += 1
        # If right packet is empty, take only from left.
        elif j == len(right):
            out.append(left[i])
            i += 1
        # If both still have cards, probability of taking from left
        # is proportional to how many remain there compare to the total number of remaining.
        else:
            p_left = (len(left) - i) / ((len(left) - i) + (len(right) - j))
            if rng.random() < p_left:
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
    # Replaces the original deck in place
    deck[:] = out


def perfect_riffle_shuffle(deck: List[int], rng: random.Random, start_side: str = "left") -> None:
    """
    Simulate a aeterministic perfect riffle / faro-style interleaving, not random at all
    This shuffle only happens if we choose the perfect shuffle at website and

    It is considered as the standard riffle shuffle,
    which back to original order after several times of shuffle, based on card size

    start_side='left' gives an out-shuffle style interleave when the two halves are equal.
    start_side='right' gives an in-shuffle style interleave when the two halves are equal.
    """
    _ = rng
    n = len(deck)
    if n <= 1:
        return

    cut = (n + 1) // 2 if start_side == "left" else n // 2
    left = deck[:cut]
    right = deck[cut:]

    out: List[int] = []
    i = j = 0
    # determines which side contributes first
    take_left = start_side == "left"

    while i < len(left) or j < len(right):
        if take_left and i < len(left):
            out.append(left[i])
            i += 1
        elif (not take_left) and j < len(right):
            out.append(right[j])
            j += 1
        elif i < len(left):
            out.append(left[i])
            i += 1
        elif j < len(right):
            out.append(right[j])
            j += 1
        take_left = not take_left

    deck[:] = out



def overhand_shuffle(deck: List[int], rng: random.Random, p: float = 0.5) -> None:
    """
    This method implement a Pemantle style overhand shuffle way for shuffle cards, it change the card's order
    This simulates that cutpoints are placed independently with probability p, then reverse each packet in place
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    n = len(deck)
    if n <= 1:
        return
    # For each gap between adjacent cards, decide if it becomes a cutpoint, which creates random packet boundaries
    cutpoints = [rng.random() < p for _ in range(n - 1)]
    # packets will store the resulting chunks, and start shows the start index of the current packet.
    packets: List[List[int]] = []
    start = 0
    # Scan through every possible cut location, if a cut happens, close the current packet and start a new one.
    for i, is_cut in enumerate(cutpoints):
        if is_cut:
            packets.append(deck[start:i + 1])
            start = i + 1
    packets.append(deck[start:])
    # Reverse each packet individuallyn and write it back into the same place in the deck.
    # The order of packet unchanged but each packet’s internal order is reversed.
    idx = 0
    for pkt in packets:
        pkt.reverse()
        deck[idx:idx + len(pkt)] = pkt
        idx += len(pkt)



def pile_shuffle(deck: List[int], rng: random.Random, k: int = 7, random_pickup: bool = True) -> None:
    """
    This method implement a pile shuffle way for shuffle cards, it change the card's order
    This distribute cards cyclically into k piles, then gather the piles.

    Parameters:
    k: the number of piles we have, we deault have 7 piles
    random_pickup: whether we gather the pile randomly
    """
    n = len(deck)
    if n <= 1 or k <= 1:
        return

    piles: List[List[int]] = [[] for _ in range(k)]
    # we distrubute cards round-robin into piles
    # card 0 to pile 0, card 1 to pile 1, ..., card k back to pile 0
    for idx, card in enumerate(deck):
        piles[idx % k].append(card)

    # shows theorder in which piles will be gathered, if random_pickup is True, then piles will be clooected randomly.
    order = list(range(k))
    if random_pickup:
        rng.shuffle(order)

    out: List[int] = []
    for pi in order:
        out.extend(piles[pi])

    deck[:] = out

# we add this to created an dishonest situation, in which the shuffle is no longer standard
# This create “biased” shuffle behaviors for shuffle experiments, simulate realistic situation
def apply_cheat(
    deck: List[int],
    mode: str = "none",
    cheat_cards: int = 0,
    tracked_card: int = 0,
    original_deck: Optional[List[int]] = None,
) -> None:
    # If no cheat requested, do nothing to the card
    if mode == "none" or cheat_cards <= 0:
        return

    cheat_cards = min(cheat_cards, len(deck))
    if cheat_cards == 0:
        return

    base = original_deck or standard_deck(len(deck))
    # we simulate three different cheat way here, for each of them,
    # if there are several cards, keep them in desire position
    # and preserve their order as in the base deck
    # move the top cards always back to top,
    if mode == "keep_top":
        forced = list(base[:cheat_cards])
        remaining = [c for c in deck if c not in set(forced)]
        deck[:] = forced + remaining
    # move the bottom cards always back to bottom
    elif mode == "keep_bottom":
        forced = list(base[-cheat_cards:])
        remaining = [c for c in deck if c not in set(forced)]
        deck[:] = remaining + forced
    # take the tracked cards from their current position and places it on top.
    elif mode == "return_tracked_to_top":
        if tracked_card in deck:
            remaining = [c for c in deck if c != tracked_card]
            deck[:] = [tracked_card] + remaining
    else:
        raise ValueError(f"Unknown cheat mode: {mode}")


# This help decide which shuffle model to use with capturing the model as a closure fn,
# and package everything into ShuffleConfig.
# We converts the parameter chosen in UI into one executable shuffle configuration object.
def build_shuffle_config(
    method: str,
    riffle_cut_p: float = 0.5,
    p_overhand: float = 0.5,
    piles_k: int = 7,
    pile_random_pickup: bool = True,
    cheat_mode: str = "none",
    cheat_cards: int = 0,
    tracked_card: int = 0,
    perfect_riffle: bool = False,
    perfect_riffle_start: str = "left",
) -> ShuffleConfig:
    params: Dict[str, object] = {}
    # When cheating, the default card order is referenced from the initial standard
    original_deck = None

    # for different shuffle ways we define different fn,
    # which call different shuffle method then apply chear mode if any
    if method == "riffle":
        if perfect_riffle:
            params = {"perfect_riffle": True, "start_side": perfect_riffle_start}

            def fn(deck: List[int], rng: random.Random) -> None:
                perfect_riffle_shuffle(deck, rng, start_side=perfect_riffle_start)
                apply_cheat(deck, cheat_mode, cheat_cards, tracked_card=tracked_card, original_deck=original_deck)

            name = f"perfect_riffle(start={perfect_riffle_start})"
        else:
            params = {"cut_p": riffle_cut_p}

            def fn(deck: List[int], rng: random.Random) -> None:
                riffle_shuffle_gsr(deck, rng, cut_p=riffle_cut_p)
                apply_cheat(deck, cheat_mode, cheat_cards, tracked_card=tracked_card, original_deck=original_deck)

            name = f"riffle(cut_p={riffle_cut_p:.2f})"
    elif method == "overhand":
        params = {"p": p_overhand}

        def fn(deck: List[int], rng: random.Random) -> None:
            overhand_shuffle(deck, rng, p=p_overhand)
            apply_cheat(deck, cheat_mode, cheat_cards, tracked_card=tracked_card, original_deck=original_deck)

        name = f"overhand(p={p_overhand:.2f})"
    elif method == "pile":
        params = {"k": piles_k, "random_pickup": pile_random_pickup}

        def fn(deck: List[int], rng: random.Random) -> None:
            pile_shuffle(deck, rng, k=piles_k, random_pickup=pile_random_pickup)
            apply_cheat(deck, cheat_mode, cheat_cards, tracked_card=tracked_card, original_deck=original_deck)

        name = f"pile(k={piles_k}, random_pickup={pile_random_pickup})"
    else:
        raise ValueError(f"Unknown method: {method}")

    if cheat_mode != "none" and cheat_cards > 0:
        name += f" + cheat[{cheat_mode}:{cheat_cards}]"

    return ShuffleConfig(
        name=name,
        method=method,
        params=params,
        shuffle_fn=fn,
        cheat_mode=cheat_mode,
        cheat_cards=cheat_cards,
        tracked_card=tracked_card,
    )


# Work as a batch shuffle configuration generator,
# returns a list of default configurations for comparison
def make_shuffle_configs(
    p_overhand: float = 0.5,
    piles_k: int = 7,
    pile_random_pickup: bool = True,
    riffle_cut_p: float = 0.5,
    cheat_mode: str = "none",
    cheat_cards: int = 0,
    tracked_card: int = 0,
    perfect_riffle: bool = False,
    perfect_riffle_start: str = "left",
) -> List[ShuffleConfig]:
    return [
        build_shuffle_config(
            method="riffle",
            riffle_cut_p=riffle_cut_p,
            cheat_mode=cheat_mode,
            cheat_cards=cheat_cards,
            tracked_card=tracked_card,
            perfect_riffle=perfect_riffle,
            perfect_riffle_start=perfect_riffle_start,
        ),
        build_shuffle_config(
            method="overhand",
            p_overhand=p_overhand,
            cheat_mode=cheat_mode,
            cheat_cards=cheat_cards,
            tracked_card=tracked_card,
        ),
        build_shuffle_config(
            method="pile",
            piles_k=piles_k,
            pile_random_pickup=pile_random_pickup,
            cheat_mode=cheat_mode,
            cheat_cards=cheat_cards,
            tracked_card=tracked_card,
        ),
    ]


### Metrics used for check randomness.
def ks_distance_to_uniform(samples: List[int], n: int) -> float:
    """
    Compute the Kolmogorov–Smirnov (KS) distance between the empirical
    distribution of tracked-card positions and the uniform distribution.

    we count how often each position appears in `samples`
    then build cumulative distribution function (CDF) of observed data and compare with uniform CDF,
    return the maximum absolute difference between the two CDFs

    Notice the result
    0.0  → perfectly uniform distribution
    >0, larger value → stronger positional bias
    """
    counts = Counter(samples)
    cum_emp = 0.0
    max_diff = 0.0
    m = len(samples)
    for x in range(n):
        cum_emp += counts.get(x, 0) / m
        cum_uni = (x + 1) / n
        max_diff = max(max_diff, abs(cum_emp - cum_uni))
    return max_diff

def tv_distance_to_uniform(samples: List[int], n: int) -> float:
    """
    Compute total variation distance between the empirical distribution
    of tracked-card positions and the uniform distribution.

    TV(P, U) = 0.5 * sum_x |P(x) - 1/n|

    Interpretation:
        0.0  -> perfectly uniform
        larger value -> more total probability mass is misplaced
    """
    if n <= 0:
        return 0.0
    if not samples:
        return 0.0

    counts = Counter(samples)
    total = len(samples)
    uniform_p = 1.0 / n
    return 0.5 * sum(abs(counts.get(x, 0) / total - uniform_p) for x in range(n))



def shannon_entropy_bits_from_counts(counts: Counter, total: int) -> float:
    """
    Measures how spread out the tracked card’s position probabilities are,
    indicating how evenly it is distributed across positions.

    we use the formula: H = -Σ p(x) * log2(p(x))
    where: p(x) = count(x) / total
    return the entropy value in [0, log2(n)]

    Notice the result
    Maximum entropy = log2(n) → perfectly uniform distribution
    Lower entropy → probability concentrated on fewer positions
    """
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h



def inversion_count(deck: List[int]) -> int:
    """
    Counts how many pairs of cards are disorder,
    which answer how far is the deck from its original sorted order
    i.e we get a inversion if have a pair (i, j) such that i < j and deck[i] > deck[j]

    return the inversion count value in [0, n(n-1)/2]

    Notice the result
    0 → fully sorted deck (no randomness)
    ~n(n-1)/4 → expected value for a random permutation
    """
    inv = 0
    n = len(deck)
    for i in range(n):
        ai = deck[i]
        for j in range(i + 1, n):
            if ai > deck[j]:
                inv += 1
    return inv



def rising_sequences(deck: List[int]) -> int:
    """
    Counts how many increasing sequences (runs) appear in the deck,
    which measure how often order is broken locally

    return the rising sequence value in [1, n]

    Notice the result
    1 → perfectly ordered deck
    ~ (n+1)/2 → expected for a random permutation
    fewer runs → too structured
    """
    if not deck:
        return 0
    runs = 1
    for i in range(1, len(deck)):
        if deck[i] < deck[i - 1]:
            runs += 1
    return runs



def mean_std(xs: List[float]) -> Tuple[float, float]:
    """
    Computes the average and variability of a metric across many trials,
    which summarize simulation results and quantify uncertainty.

    we use formula: mean = Σ x / n
                    variance = Σ (x - mean)^2 / (n - 1)

    return the mean and variance value

    Notice the result
    mean → central tendency
    std → variability / noise level
    """
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)



def uniform_inversion_baseline(n: int) -> Tuple[float, float]:
    """
    This provides the theoretical expected value (and variance) of inversion count
    for a perfectly random deck, served as a benchmark

    """
    e = n * (n - 1) / 4
    var = n * (n - 1) * (2 * n + 5) / 72
    return e, math.sqrt(var)



def expected_rising_sequences_uniform(n: int) -> float:
    """
    This provides the theoretical expected value of rising_sequences
    for a perfectly random deck, served as a benchmark
    """
    return (n + 1) / 2


# THis function takes raw simulation results from many trials
# and turns them into summary metrics for the app.
def summarize_step(
    pos_samples: List[int],
    inv_samples: List[int],
    run_samples: List[int],
    deck_size: int,
    card_position_counts: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, object]:
    trials_done = len(pos_samples)
    inv_u_mean, inv_u_sd = uniform_inversion_baseline(deck_size)
    runs_uniform_mean = expected_rising_sequences_uniform(deck_size)

    pos_mean, pos_sd = mean_std([float(x) for x in pos_samples])
    inv_mean, inv_sd = mean_std([float(x) for x in inv_samples])
    runs_mean, runs_sd = mean_std([float(x) for x in run_samples])

    pos_counts = Counter(pos_samples)
    pos_entropy = shannon_entropy_bits_from_counts(pos_counts, trials_done)
    pos_entropy_max = math.log2(deck_size)

    se_scale = math.sqrt(trials_done) if trials_done > 0 else 1.0
    return {
        "trials_done": trials_done,
        "ks_pos": ks_distance_to_uniform(pos_samples, n=deck_size),
        "ks_pos_benchmark": 0.0,
        "tv_pos": tv_distance_to_uniform(pos_samples, n=deck_size),
        "tv_pos_benchmark": 0.0,
        "pos_mean": pos_mean,
        "pos_mean_benchmark": (deck_size - 1) / 2,
        "pos_sd": pos_sd,
        "pos_se": pos_sd / se_scale,
        "pos_entropy": pos_entropy,
        "pos_entropy_frac": pos_entropy / pos_entropy_max if pos_entropy_max > 0 else 1.0,
        "pos_entropy_frac_benchmark": 1.0,
        "inv_mean": inv_mean,
        "inv_sd": inv_sd,
        # standard error which measure how uncertain the mean estimate is
        "inv_se": inv_sd / se_scale,
        "inv_uniform_mean": inv_u_mean,
        "inv_uniform_sd": inv_u_sd,
        "runs_mean": runs_mean,
        "runs_mean_benchmark": runs_uniform_mean,
        "runs_sd": runs_sd,
        "runs_se": runs_sd / se_scale,
        "pos_counts": dict(pos_counts),
        "card_position_counts": {int(card): {int(pos): int(cnt) for pos, cnt in pos_map.items()} for card, pos_map in (card_position_counts or {}).items()},
    }


# Schedules for each method to run
#  which indicates the step numbers chosen by the shuffle method to evaluate
def schedule_riffle(max_shuffles: int = 12) -> List[int]:
    return list(range(1, max_shuffles + 1))


def schedule_overhand(long: bool = False) -> List[int]:
    # overhand mixing is much slower, roughly n^2*log(n), so we might need large value
    if long:
        return [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000, 15000, 20000, 30000]
    return [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]


def schedule_pile(max_steps: int = 200, stride: int = 5) -> List[int]:
    # pile shuffle behavior can change sharply at first,
    # then we need fine early resolution and coarser later resolution.
    dense_early = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50]
    more = list(range(60, max_steps + 1, stride))
    return sorted({s for s in dense_early + more if 1 <= s <= max_steps})


### helpers for batch simulation

# this functino do one batch of Monte Carlo experiments efficiently,
# which actually simulates many independent trials, shuffles decks, records tracked-card positions, inversion counts,
# run counts, and per-card position frequencies at the requested step
def _simulate_trial_batch(
    cfg: ShuffleConfig,
    steps_list: List[int],
    batch_trials: int,
    seed: int,
    tracked_card: int,
    deck_size: int,
) -> Dict[int, Dict[str, List[int]]]:
    # Master RNG(Random Number Generator) for reproducible sub-seeds.
    rng_master = random.Random(seed)
    # Sort requested checkpoints and get the furthest one.
    steps_list = sorted(steps_list)
    max_steps = steps_list[-1]

    # Prepare storage for every checkpoint.
    pos_samples = {s: [] for s in steps_list}
    inv_samples = {s: [] for s in steps_list}
    run_samples = {s: [] for s in steps_list}
    card_position_counts = {s: {card: Counter() for card in range(deck_size)} for s in steps_list}

    for _ in range(batch_trials):
        # we get Independent RNG per trial which start from ordered deck
        rng = random.Random(rng_master.randrange(1 << 30))
        deck = standard_deck(deck_size)
        next_idx = 0
        target = steps_list[next_idx]
        # Apply one shuffle step at a time, loop through all checkpoints and collect their data
        for step in range(1, max_steps + 1):
            cfg.shuffle_fn(deck, rng)
            # at each step targeted, record tracked-card position, count inversion and rising-sequence
            if step == target:
                pos_samples[target].append(deck.index(tracked_card))
                inv_samples[target].append(inversion_count(deck))
                run_samples[target].append(rising_sequences(deck))
                for position, card in enumerate(deck):
                    card_position_counts[target][card][position] += 1
                next_idx += 1
                if next_idx >= len(steps_list):
                    break
                target = steps_list[next_idx]

    return {
        s: {"pos": pos_samples[s], "inv": inv_samples[s], "runs": run_samples[s], "card_position_counts": card_position_counts[s]}
        for s in steps_list
    }


# calls run_trials_live and stores each yielded partial result in final, then returns the last one after loop
def run_trials(
    cfg: ShuffleConfig,
    steps_list: List[int],
    trials: int = 500,
    seed: int = 123,
    tracked_card: int = 0,
    deck_size: int = 52,
    max_workers: int = 1,
) -> Dict[int, Dict[str, object]]:
    final = None
    for final in run_trials_live(
        cfg=cfg,
        steps_list=steps_list,
        trials=trials,
        seed=seed,
        tracked_card=tracked_card,
        deck_size=deck_size,
        batch_size=trials,
        max_workers=max_workers,
    ):
        pass
    return final or {}


# This is a general engine, which runs simulations in batches, optionally in parallel,
# merges the outputs, and yields progressively improved estimates.
def run_trials_live(
    cfg: ShuffleConfig,
    steps_list: List[int],
    trials: int = 500,
    seed: int = 123,
    tracked_card: int = 0,
    deck_size: int = 52,
    batch_size: int = 100,
    max_workers: int = 1,
) -> Iterable[Dict[int, Dict[str, object]]]:
    """
    This is a general engine, live Monte Carlo generator,
    which repeatedly runs chunks of shuffle trials, merges the raw data,
    computes updated randomness statistics,
    and yields those partial results so the Streamlit app can refresh while the experiment is still running.

    :param cfg:This indicates chosen shuffle experiment: riffle, overhand, pile, perfect riffle, cheat mode, and so on.
    :param steps_list: These are the checkpoint steps where you want measurements,
    :param trials:Total number of Monte Carlo runs to perform
    :param seed:Base random seed so the simulation is reproducible
    :param tracked_card:The card whose position distribution we are tracking.
    :param deck_size:Number of cards in the deck.
    :param batch_size: The function do trails in chunks of this size.
    :param max_workers: Maximum number of worker threads used inside a batch.

    It produces a sequence of result dictionaries over time by using "yield".
    """
    # make sure the requested measurement steps are valid, unique, and ordered before any simulation starts
    steps_list = sorted(set(int(s) for s in steps_list))
    if not steps_list:
        return

    # create the master storage that will accumulate results batch after batch,
    # instead of discarding earlier work
    agg_pos = {s: [] for s in steps_list}
    agg_inv = {s: [] for s in steps_list}
    agg_runs = {s: [] for s in steps_list}
    agg_card_pos = {s: {card: Counter() for card in range(deck_size)} for s in steps_list}

    # control the main batch loop, help generate distinct seeds for different batches.
    remaining = trials
    batch_index = 0

    while remaining > 0:
        # control the main batch loop, help generate distinct seeds for different batches
        # and distribute work as evenly as possible among the threads.
        current_batch_size = min(batch_size, remaining)
        worker_count = max(1, min(max_workers, current_batch_size))
        worker_sizes = [current_batch_size // worker_count] * worker_count
        for i in range(current_batch_size % worker_count):
            worker_sizes[i] += 1

        # convert the abstract split into actual runnable tasks with separate seeds
        jobs = []
        offset = 0
        for size in worker_sizes:
            if size <= 0:
                continue
            jobs.append((size, seed + 10007 * batch_index + offset))
            offset += 1
        # This runs the actual simulation work by calling _simulate_trial_batch
        # If there is only one job, no need for thread overhead
        if len(jobs) == 1:
            batch_outputs = [
                _simulate_trial_batch(cfg, steps_list, jobs[0][0], jobs[0][1], tracked_card, deck_size)
            ]
        else:
            # Create a thread pool with one thread per job
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                futures = [
                    # Submit each sub-batch to the pool and
                    ex.submit(_simulate_trial_batch, cfg, steps_list, size, sub_seed, tracked_card, deck_size)
                    for size, sub_seed in jobs
                ]
                batch_outputs = [f.result() for f in futures]

        # combine all worker results so the simulation behaves as one huge batch had been run.
        for out in batch_outputs:
            for s in steps_list:
                agg_pos[s].extend(out[s]["pos"])
                agg_inv[s].extend(out[s]["inv"])
                agg_runs[s].extend(out[s]["runs"])
                for card, counter in out[s]["card_position_counts"].items():
                    agg_card_pos[s][card].update(counter)

        # THis turn raw Monte Carlo samples into the polished statistics the app can plot.
        # Build one summary dictionary for each checkpoint step.
        current_results = {
            s: summarize_step(
                agg_pos[s],
                agg_inv[s],
                agg_runs[s],
                deck_size,
                card_position_counts={card: dict(counter) for card, counter in agg_card_pos[s].items()},
            )
            for s in steps_list
        }
        # sends the current batch-accumulated results back to the caller without ending the function
        yield current_results

        remaining -= current_batch_size
        batch_index += 1


### helpers for plotting(visualization)
def results_to_frame(results):
    """
    This converts nested result dictionaries into row-wise records suitable for pandas DataFrames
    """
    rows = []
    for step in sorted(results):
        row = {"steps": step}

        for k, v in results[step].items():
            if k == "pos_counts" or k == "card_position_counts":
                # convert dict safely
                if isinstance(v, dict):
                    v = json.dumps(v)  # ← FIX HERE
            row[k] = v

        rows.append(row)

    return rows


def all_card_extreme_probability_rows(results: Dict[int, Dict[str, object]], deck_size: int) -> List[Dict[str, float]]:
    """
    For each card and step, find its most and least likely position,
    which provides support for the “top-bottom” diagnostic bars showing where
    each card is unusually concentrated or unusually absent.

    """
    rows: List[Dict[str, float]] = []
    for step in sorted(results):
        all_counts = results[step].get("card_position_counts", {})
        total = results[step].get("trials_done", 0) or 1
        for card in range(deck_size):
            pos_counts = all_counts.get(card, {})
            if not pos_counts:
                max_pos = 0
                min_pos = 0
                max_prob = 0.0
                min_prob = 0.0
            else:
                probs = {pos: pos_counts.get(pos, 0) / total for pos in range(deck_size)}
                max_pos = max(probs, key=lambda pos: (probs[pos], -pos))
                min_pos = min(probs, key=lambda pos: (probs[pos], pos))
                max_prob = probs[max_pos]
                min_prob = probs[min_pos]
            rows.append({
                "steps": step,
                "card": card,
                "extreme_type": "Highest position probability",
                "position": max_pos,
                "probability": max_prob,
            })
            rows.append({
                "steps": step,
                "card": card,
                "extreme_type": "Lowest position probability",
                "position": min_pos,
                "probability": min_prob,
            })
    return rows


def position_probability_rows(results: Dict[int, Dict[str, object]], deck_size: int) -> List[Dict[str, float]]:
    """
    This converts tracked-card position counts into probabilities per position, per step.
    """
    rows: List[Dict[str, float]] = []
    for step in sorted(results):
        counts = results[step].get("pos_counts", {})
        total = results[step].get("trials_done", 0) or 1
        for pos in range(deck_size):
            rows.append(
                {
                    "steps": step,
                    "position": pos,
                    "count": counts.get(pos, 0),
                    "probability": counts.get(pos, 0) / total,
                }
            )
    return rows


def deterministic_deck_path(cfg: ShuffleConfig, steps: int, deck_size: int = 52) -> List[Dict[str, object]]:
    """
    This apply a deterministic shuffle repeatedly and store the whole deck state after each step,
    especially used for perfect riffle exploration
    """
    deck = standard_deck(deck_size)
    rng = random.Random(0)
    rows: List[Dict[str, object]] = []
    for step in range(0, steps + 1):
        rows.append(
            {
                "steps": step,
                "runs": rising_sequences(deck),
                "inversions": inversion_count(deck),
                "deck": deck.copy(),
                "returned_to_original": deck == standard_deck(deck_size),
            }
        )
        if step < steps:
            cfg.shuffle_fn(deck, rng)
    return rows


def find_return_to_original(cfg: ShuffleConfig, deck_size: int = 52, max_steps: int = 500) -> Optional[int]:
    """
    This repeatedly apply the chosen deterministic shuffle and return the first step at which the deck cycles back,
    which is nice way to show periodicity of perfect riffles.
    """
    deck = standard_deck(deck_size)
    original = deck.copy()
    rng = random.Random(0)
    for step in range(1, max_steps + 1):
        cfg.shuffle_fn(deck, rng)
        if deck == original:
            return step
    return None


### Timing benchmark
def benchmark_shuffle_step(
    cfg: ShuffleConfig,
    steps: int = 1000,
    repeats: int = 200,
    seed: int = 1,
    deck_size: int = 52,
) -> float:
    """
    This function measures how fast a shuffle method is to run in code,
    which measures computation time per shuffle step,
    helps compare whether riffle, overhand, or pile shuffle is faster or slower to simulate
    """
    rng = random.Random(seed)
    deck = standard_deck(deck_size)

    t0 = time.perf_counter()
    for _ in range(repeats):
        deck[:] = standard_deck(deck_size)
        for _ in range(steps):
            cfg.shuffle_fn(deck, rng)
    t1 = time.perf_counter()

    total_ops = repeats * steps
    return (t1 - t0) / total_ops


### Demo locally
def _print_progress_line(trials: int, step: int, r: Dict[str, object]) -> None:
    """It takes one result dictionary and prints a compact progress report with:
    number of trials, step number, tracked-card KS distance, mean inversion count, mean rising runs."""
    print(
        f"trials={trials:4d} | step={step:3d} | "
        f"KS={r['ks_pos']:.3f} | inv_mean={r['inv_mean']:.1f} | runs_mean={r['runs_mean']:.2f}"
    )



def demo_three_shuffle_methods_explicit() -> None:
    """
    It runs three shuffle types as riffle, overhand, pile,
    and shows how their results look for several trial counts, then prints timing comparisons
    """
    tracked_card = 0
    seed = 2026
    trials_list = [200, 800, 2000]

    # each entry has: a short method label, a built shuffle configuration and a list of checkpoint steps.
    method_runs = [
        ("riffle", build_shuffle_config(method="riffle", riffle_cut_p=0.5), [1, 2, 3, 5, 7, 8, 10, 12]),
        ("overhand", build_shuffle_config(method="overhand", p_overhand=0.5), [10, 20, 50, 100, 196]),
        ("pile", build_shuffle_config(method="pile", piles_k=7, pile_random_pickup=True), [1, 2, 3, 5, 10, 20, 30, 50, 80, 120, 160, 196]),
    ]

    print("\n=== Explicit progress for three shuffle methods ===")
    for method_name, cfg, steps_list in method_runs:
        final_step = steps_list[-1]
        print(f"\n=== {method_name}: {cfg.name} ===")
        for trials in trials_list:
            res = run_trials(
                cfg=cfg,
                steps_list=steps_list,
                trials=trials,
                seed=seed,
                tracked_card=tracked_card,
            )
            _print_progress_line(trials, final_step, res[final_step])

    print("\n=== Timing (seconds per shuffle application) ===")
    for _, cfg, _ in method_runs:
        sec_per = benchmark_shuffle_step(cfg, steps=2000, repeats=200)
        print(f"{cfg.name:40s} : {sec_per:.3e} sec/shuffle")



def write_rows_to_csv(rows: List[Dict[str, object]], csv_path: str) -> None:
    """
    This saves a list of result dictionaries into a CSV file
    """
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def export_results_csv(results: Dict[int, Dict[str, object]], csv_path: str) -> None:
    """
    This function first converts results into rows then writes them to csv_path
    """
    write_rows_to_csv(results_to_frame(results), csv_path)



def demo_sweep_piles_and_riffle_bias(
    write_csv: bool = True,
    csv_prefix: str = "",
    max_workers: int = 1,
) -> None:
    """
    this function is used for exploring how changing parameters changes the behavior of the shuffle models.
    """
    trials_list = [200, 800, 2000]
    tracked_card = 0
    seed = 2026

    for k in [4, 7, 10, 13, 26, 52]:
        cfg = make_shuffle_configs(piles_k=k)[2]
        steps_list = [1, 2, 3, 5, 10, 20, 30, 50, 80, 120, 160, 196]
        print(f"\n=== {cfg.name} ===")
        for trials in trials_list:
            res = run_trials(
                cfg,
                steps_list,
                trials=trials,
                seed=seed,
                tracked_card=tracked_card,
                max_workers=max_workers,
            )
            if write_csv:
                csv_name = CSV_DIR / f"{csv_prefix}pile_k_{k}_trials_{trials}.csv"
                export_results_csv(res, csv_name)
            print(f"-- trials={trials} --")
            for s in steps_list:
                _print_progress_line(trials, s, res[s])

    for cut_p in [0.5, 0.6, 0.7, 0.8]:
        cfg = make_shuffle_configs(riffle_cut_p=cut_p)[0]
        steps_list = schedule_riffle(12)
        print(f"\n=== {cfg.name} ===")
        res = run_trials(
            cfg,
            steps_list,
            trials=1000,
            seed=seed,
            tracked_card=tracked_card,
            max_workers=max_workers,
        )
        for s in steps_list:
            r = res[s]
            print(
                f"steps={s:2d} | KS={r['ks_pos']:.3f} | "
                f"Hpos={r['pos_entropy']:.3f} ({100 * r['pos_entropy_frac']:.1f}%) | "
                f"inv_mean={r['inv_mean']:.1f}"
            )
    # picks the overhand configuration and perform slow mixing
    cfg = make_shuffle_configs(p_overhand=0.5)[1]
    steps_list = schedule_overhand(long=True)
    print(f"\n=== {cfg.name} (long schedule) ===")
    res = run_trials(
        cfg,
        steps_list,
        trials=1200,
        seed=seed,
        tracked_card=tracked_card,
        max_workers=max_workers,
    )
    for s in steps_list:
        r = res[s]
        print(
            f"steps={s:5d} | KS={r['ks_pos']:.3f} | Hpos={r['pos_entropy']:.3f} | "
            f"inv_mean={r['inv_mean']:.1f} | runs_mean={r['runs_mean']:.2f}"
        )

    print("\n=== Timing (seconds per shuffle application) ===")
    # builds one representative config for each shuffle family.
    cfgs = [
        make_shuffle_configs(riffle_cut_p=0.5)[0],
        make_shuffle_configs(p_overhand=0.5)[1],
        make_shuffle_configs(piles_k=7)[2],
    ]
    for c in cfgs:
        sec_per = benchmark_shuffle_step(c, steps=2000, repeats=200)
        print(f"{c.name:40s} : {sec_per:.3e} sec/shuffle")


if __name__ == "__main__":
    demo_sweep_piles_and_riffle_bias()
