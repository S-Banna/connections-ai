import numpy as np
from itertools import combinations
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import random


# LOADING DATASET

import json

with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("Loaded games:", len(dataset))
print("Sample game keys:", dataset[0].keys())

# LOADING FASTTEXT

def load_fasttext(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8", newline='\n', errors='ignore') as f:
        first_line = next(f)
        total = int(first_line.split()[0])  # number of words

        for line in tqdm(f, total=total):
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            embeddings[word] = vec
    return embeddings

ft = load_fasttext("wiki-news-300d-1M-subword.vec/wiki-news-300d-1M-subword.vec")


# HELPER FUNCTIONS

def normalize(word):
    return word.lower().replace("'", "").replace("-", " ")

def get_vector(word):
    return ft.get(normalize(word), None)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# TRAINING

split_idx = int(len(dataset) * 0.05)

train_games = dataset[split_idx:]
test_games = dataset[:split_idx]
manual_testing = test_games[:10]

X = []
y = []

for game in train_games:
    correct_groups = game["answers"]
    all_words = game["words"]

    correct_sets = [set(g["words"]) for g in correct_groups]

    # positives
    for g in correct_groups:
        vecs = [get_vector(w) for w in g["words"]]
        if any(v is None for v in vecs):
            continue

        mean = np.mean(vecs, axis=0)

        pairs = list(combinations(vecs, 2))
        sims = [cosine(a, b) for a, b in pairs]

        sim_mean = np.mean(sims)
        sim_std = np.std(sims)

        feat = np.concatenate([mean, [sim_mean, sim_std]])
        X.append(feat)

        y.append(1)

    # negatives
    all_combos = list(combinations(all_words, 4))
    random.shuffle(all_combos)
    
    neg_added = 0
    for combo in all_combos:
        if set(combo) not in correct_sets:
            vecs = [get_vector(w) for w in combo]
            if any(v is None for v in vecs):
                continue
            mean = np.mean(vecs, axis=0)

            pairs = list(combinations(vecs, 2))
            sims = [cosine(a, b) for a, b in pairs]

            sim_mean = np.mean(sims)
            sim_std = np.std(sims)

            feat = np.concatenate([mean, [sim_mean, sim_std]])
            X.append(feat)

            y.append(0)
            neg_added += 1
        if neg_added >= 10:
            break

model = MLPClassifier(
    hidden_layer_sizes=(64,),
    max_iter=10
)

print("Training samples:", len(X))
model.fit(X, y)


# SCORING

def nn_score(group):
    vecs = [get_vector(w) for w in group]
    if any(v is None for v in vecs):
        return 0

    mean = np.mean(vecs, axis=0)

    pairs = list(combinations(vecs, 2))
    sims = [cosine(a, b) for a, b in pairs]

    sim_mean = np.mean(sims)
    sim_std = np.std(sims)

    feat = np.concatenate([mean, [sim_mean, sim_std]])

    return model.predict_proba([feat])[0][1]

def final_score(group):
    # cosine part
    vecs = [get_vector(w) for w in group]
    if any(v is None for v in vecs):
        return -1

    pairs = list(combinations(vecs, 2))
    sims = [cosine(a, b) for a, b in pairs]
    cos_score = sum(sims) / len(sims)

    # nn part
    nn_s = nn_score(group)

    # weighting
    return 0.5 * cos_score + 0.5 * nn_s


# SAMPLE

test_game = random.choice(manual_testing)

print("\nWords:", test_game["words"])
print("\nCorrect:")
for g in test_game["answers"]:
    print(g["words"])

all_groups = list(combinations(test_game["words"], 4))

scored = []
for g in all_groups:
    s = final_score(g)
    if s != -1:
        scored.append((g, s))

scored.sort(key=lambda x: x[1], reverse=True)

correct_sets = [set(g["words"]) for g in test_game["answers"]]

print("\nTop guesses:")
for i, (g, s) in enumerate(scored[:10]):
    print(i+1, set(g) in correct_sets, f"{s:.3f}", g)