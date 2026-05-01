# generate_dataset.py
# Synthetic PPRL dataset: Party A (clean) and Party B (corrupted), plus labeled pair file.
#
# INSTALL: pip install pandas numpy faker
#
# OUTPUT (written to output/):
#   party_a.csv  Clean records
#   party_b.csv  Corrupted records (same people as A, same row order)
#   pairs.csv    Labeled pairs: 50k matches + 450k random non-matches (1:9)

import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Settings ─────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
NUM_PEOPLE      = 50_000
NONMATCH_RATIO  = 9        # 1:9 positive ratio
CORRUPTION_RATE = 0.30     # probability any given field gets corrupted
DROP_EMAIL_RATE = 0.20     # records with email entirely missing
DROP_PHONE_RATE = 0.25     # records with phone entirely missing

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)
fake = Faker("en_US")

RECORD_COLS = [
    "first_name", "last_name", "street_address",
    "city", "state", "zip_code",
    "date_of_birth", "email", "phone",
]

# ── Step 1: Generate clean records (Party A) ─────────────────────────────
print(f"Generating {NUM_PEOPLE:,} clean records...")

party_a = pd.DataFrame([{
    "first_name":     fake.first_name(),
    "last_name":      fake.last_name(),
    "street_address": fake.street_address(),
    "city":           fake.city(),
    "state":          fake.state_abbr(),
    "zip_code":       fake.zipcode(),
    "date_of_birth":  fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%Y-%m-%d"),
    "email":          fake.email(),
    "phone":          fake.phone_number(),
} for _ in range(NUM_PEOPLE)], columns=RECORD_COLS)

# ── Step 2: Corrupt records to produce Party B ───────────────────────────
# Each field has CORRUPTION_RATE chance of one edit per call. Two edit functions:
# text fields get character-level edits, digit-bearing fields get an adjacent-digit swap.

def corrupt_text(value):
    if not value or random.random() > CORRUPTION_RATE:
        return value
    action = random.choice(["substitute", "delete", "blank"])
    if action == "blank":
        return ""
    i = random.randint(0, len(value) - 1)
    if action == "substitute":
        return value[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + value[i + 1:]
    return value[:i] + value[i + 1:]   # delete

def corrupt_digits(value):
    if not value or random.random() > CORRUPTION_RATE:
        return value
    digit_positions = [i for i, c in enumerate(value) if c.isdigit()]
    if len(digit_positions) < 2:
        return value
    i = random.choice(digit_positions[:-1])
    chars = list(value)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)

print("Corrupting records for Party B...")
party_b = party_a.copy()
for col in ["first_name", "last_name", "street_address", "city", "email"]:
    party_b[col] = party_b[col].apply(corrupt_text)
for col in ["zip_code", "phone", "date_of_birth"]:
    party_b[col] = party_b[col].apply(corrupt_digits)

# Some records have entire fields missing (independent of typo corruption above)
print("Dropping some fields in Party B...")
rng_missing = np.random.default_rng(RANDOM_SEED + 3)
party_b.loc[rng_missing.random(NUM_PEOPLE) < DROP_EMAIL_RATE, "email"] = ""
party_b.loc[rng_missing.random(NUM_PEOPLE) < DROP_PHONE_RATE, "phone"] = ""

# ── Step 3: Build candidate pair file ────────────────────────────────────
# Matches are positional: row i in A ↔ row i in B.
# `is_match` is not stored; load_candidates() in utils.py derives it as
# `rec_id_a == rec_id_b`, since that's what positional matching means.
matches = pd.DataFrame({
    "rec_id_a": np.arange(NUM_PEOPLE),
    "rec_id_b": np.arange(NUM_PEOPLE),
})

# Non-matches: random pairs of different people at 1:NONMATCH_RATIO.
n_non_matches = NUM_PEOPLE * NONMATCH_RATIO
rng_nm = np.random.default_rng(RANDOM_SEED + 1)
nm_a = rng_nm.choice(NUM_PEOPLE, size=n_non_matches * 2)
nm_b = rng_nm.choice(NUM_PEOPLE, size=n_non_matches * 2)
keep = nm_a != nm_b
non_matches = pd.DataFrame({
    "rec_id_a": nm_a[keep][:n_non_matches],
    "rec_id_b": nm_b[keep][:n_non_matches],
})

# Preserve the illustrative non-match pair referenced by EXAMPLE_PAIRS in Notebooks/utils.py.
ILLUSTRATIVE_NON_MATCH = {"rec_id_a": 47135, "rec_id_b": 39460}
already_present = (
    (non_matches["rec_id_a"] == ILLUSTRATIVE_NON_MATCH["rec_id_a"]) &
    (non_matches["rec_id_b"] == ILLUSTRATIVE_NON_MATCH["rec_id_b"])
).any()
if not already_present:
    non_matches = pd.concat(
        [non_matches, pd.DataFrame([ILLUSTRATIVE_NON_MATCH])],
        ignore_index=True,
    )

pairs = pd.concat([matches, non_matches], ignore_index=True)

# ── Step 4: Save ─────────────────────────────────────────────────────────
party_a.to_csv(OUTPUT_DIR / "party_a.csv", index_label="rec_id_a")
party_b.to_csv(OUTPUT_DIR / "party_b.csv", index_label="rec_id_b")
pairs.to_csv(  OUTPUT_DIR / "pairs.csv",   index=False)

print(f"""
Done. Files written to {OUTPUT_DIR}/
  party_a.csv  {NUM_PEOPLE:,} clean records
  party_b.csv  {NUM_PEOPLE:,} corrupted records
  pairs.csv    {len(pairs):,} labeled pairs ({NUM_PEOPLE:,} matches + {len(non_matches):,} non-matches, 1:{NONMATCH_RATIO})
""")
