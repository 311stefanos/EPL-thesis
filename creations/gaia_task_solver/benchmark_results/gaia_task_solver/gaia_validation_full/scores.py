from pathlib import Path
import jsonlines

per_level_score = {1: 0, 2: 0, 3: 0}
per_level_sum = {1: 0, 2: 0, 3: 0}
actual_total = {1: 53, 2: 86, 3: 26}

scores_file = Path(__file__).parent / "scored_results.jsonl"

with jsonlines.open(scores_file, 'r') as reader:
    for obj in reader.iter(type=dict):
        score = obj['score']
        level = obj['level']
        if score:
            per_level_score[level] += 1

        per_level_sum[level] += 1

for level in [1, 2, 3]:
    print(f"Level {level} ({actual_total[level]}): {per_level_score[level]}/{per_level_sum[level]} = {per_level_score[level] / per_level_sum[level] * 100:.2f}%")
