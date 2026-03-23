from bs4 import BeautifulSoup
import random, json

with open("'Connections' answer archive.htm", 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

games = []

for p in soup.find_all('p'):
    text = p.get_text(separator="\n").strip()

    if not text.startswith("Connections"):
        continue

    lines = text.split("\n")
    
    if (len(lines) < 5):
        continue

    categories = []
    all_words = []

    for line in lines[1:]:
        if ":" not in line:
            continue

        category, wordlist = line.split(":", 1)
        words = [w.strip() for w in wordlist.split(",")]

        categories.append({
            "category": category.strip(),
            "words": words
        })

        all_words.extend(words)

    random.shuffle(all_words)
    games.append({
        "words": all_words,
        "answers": categories
    })

with open("dataset.json", 'w', encoding='utf-8') as f:
    json.dump(games, f, indent=4, ensure_ascii=False)

print(f"Parsed {len(games)} games.")