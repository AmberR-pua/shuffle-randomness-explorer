# Shuffle Randomness Explorer

This is a small project I built to study how different card shuffling methods actually mix a deck.

I wanted to explore a simple but interesting question:

> When can we say a deck of cards is “random enough” under shuffling?

Instead of just shuffling cards and assuming it works, this project simulates different shuffle methods, measures how randomness evolves, and visualizes the results in an interactive way.

---

## What this project does

This is a Streamlit app where you can:

* simulate different shuffle methods (riffle, overhand, pile)
* track how a specific card moves over time
* compare multiple shuffle configurations side by side
* see how randomness improves as the number of shuffles increases
* export results for further analysis

The goal is not just to visualize shuffling, but to **understand randomness from different perspectives**.

---

## Shuffle methods implemented

* **Riffle shuffle (GSR-style)**
  A probabilistic model where the deck is split and interleaved.

* **Overhand shuffle**
  The deck is cut into random chunks and reversed.

* **Pile shuffle**
  Cards are distributed into piles and then collected.

* **Perfect riffle (deterministic)**
  A special case where the shuffle is not random at all.
  This is useful to show that some “shuffles” can still have strong structure.

---

## Metrics used

There is no single way to measure randomness, so I used multiple metrics:

### 1. KS distance (tracked card)

Measures how far the tracked card’s position distribution is from uniform.

* 0 → perfectly uniform
* higher → more bias

---

### 2. Position entropy

Measures how evenly the tracked card spreads across positions.

* closer to 1 → better spread
* lower → concentrated in certain positions

---

### 3. Inversion count

Counts how many pairs of cards are out of order.

* 0 → completely ordered
* ~n(n−1)/4 → expected for random deck

---

### 4. Rising runs

Counts how many increasing sequences exist in the deck.

* fewer runs → more structure remains
* around (n+1)/2 → expected for random deck

---

### 5. Mean position of tracked card

Checks whether the tracked card tends to stay near top or bottom.

* expected value → middle of the deck

---

## Key idea

Different metrics measure different things.

A shuffle can:

* look random for one card
* but still keep structure in the whole deck

This project tries to show that clearly.

---

## Features

* interactive UI built with Streamlit
* compare multiple shuffle settings at once
* real-time metric plots over shuffle steps
* histogram of tracked card position
* 3D visualization (step × position × probability)
* “cheat mode” to simulate biased shuffling
* perfect riffle analysis (can return to original order)
* export results as CSV
* save/load experiment configs (JSON)

---

## Tech stack

* **Python**
* **Streamlit** (UI)
* **Plotly** (visualization)
* **Pandas** (data processing)

---

## Implementation details

This project is based on **Monte Carlo simulation**.

Some techniques used:

* repeated random trials to estimate distributions
* `@st.cache_data` to avoid recomputation
* `ThreadPoolExecutor` for parallel simulation
* `dataclass` for configuration management
* converting results into DataFrames for plotting
* modular design (simulation vs UI separation)

---

## How to run this project

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/shuffle-randomness-explorer.git
cd shuffle-randomness-explorer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## Project structure

```
.
├── app.py                  # Streamlit UI
├── shuffle_main.py         # simulation + metrics
├── requirements.txt    
├── shuffle_run_local.txt   # generated local results
├── README.md
└── csv_file/               # generated results
```

---

## Why I built this

I wanted to build something that is:

* simple to understand (cards)
* but still involves probability and simulation
* interactive instead of static
* and easy to experiment with

It also helped me practice:

* structuring a small project
* thinking about how to measure “randomness”
* separation between simulation logic and UI
* build UI from users' perspectives


---

## Limitations

* simulation can be slow for very large trials
* inversion count is O(n²), so not optimized for huge decks
* randomness depends on pseudo-random generator

---

## Possible improvements

* faster algorithms for large decks
* more shuffle models
* better statistical confidence intervals
* deploy the app online

---

## Final note

This is a project about combining mathematical knowledge with coding skill, 
involving multiple aspects such as **probability, simulation, and visualization**.

---

Thanks for reading!
