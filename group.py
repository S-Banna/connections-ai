import numpy as np
from itertools import combinations
from tqdm import tqdm

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
words = [
    "Gaggle", "Pack", "Pod", "Pride",
    "Glacier", "Molasses", "Sloth", "Traffic",
    "Cartwright", "Two", "Wrath", "Wrestle",
    "Any", "Emmy", "Envy", "Okay"
]

def normalize(word):
    return word.lower().replace("'", "").replace("-", " ")

def get_vector(word):
    return ft.get(normalize(word), None)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def group_score(group):
    vecs = [get_vector(w) for w in group]
    if any(v is None for v in vecs):
        return -1  # discard invalid groups
    
    pairs = list(combinations(vecs, 2))
    sims = [cosine(a, b) for a, b in pairs]
    return sum(sims) / len(sims)

all_groups = list(combinations(words, 4))

scored = []
for g in all_groups:
    score = group_score(g)
    if score != -1:
        scored.append((g, score))

scored.sort(key=lambda x: x[1], reverse=True)

TOP_K = 10

for i, (group, score) in enumerate(scored[:TOP_K]):
    print(f"{i+1}. Score: {score:.3f} | {group}")