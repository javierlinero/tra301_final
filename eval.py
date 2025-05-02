import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

gold_pth = r"C:\Users\JL211\Desktop\evaluation_results.csv"
pred_pth = r"C:\Users\JL211\Desktop\evaluation_results_tagged.csv"

def extract_tags(text):
    tags = re.findall(r'<BACKGROUND>|<CONTRIBUTION>', text)
    return [0 if t == '<BACKGROUND>' else 1 for t in tags]

gold = pd.read_csv(gold_pth)
pred = pd.read_csv(pred_pth)

gold_dict = dict(zip(gold["FILENAME"], gold["input_text"].fillna("")))
pred_dict = dict(zip(pred["FILENAME"], pred["input_text"].fillna("")))

gold_tags = {fn: extract_tags(txt) for fn, txt in gold_dict.items()}
pred_tags = {fn: extract_tags(txt) for fn, txt in pred_dict.items()}

results = defaultdict(lambda: {'correct': 0, 'total': 0})
for fn in sorted(gold_tags):
    gt = gold_tags[fn]
    pt = pred_tags.get(fn, [])
    if len(gt) != len(pt):
        continue
    correct = sum(g == p for g, p in zip(gt, pt))
    total = len(gt)
    results[fn]['correct'] = correct
    results[fn]['total'] = total

docs = []
scores = []
for fn, vals in results.items():
    docs.append(fn)
    total = vals['total']
    correct = vals['correct']
    score = correct / total if total > 0 else 0
    scores.append(score)

norm = Normalize(vmin=0, vmax=1)
cmap = cm.get_cmap('viridis')
colors = [cmap(norm(s)) for s in scores]
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(docs, scores, color=colors)

avg = sum(scores) / len(scores)
ax.axhline(y=avg, color='red', linestyle='--', linewidth=2,
           label=f'Average = {avg:.3f}')
ax.legend()

ax.set_xlabel('Document')
ax.set_ylabel('Score')
ax.set_title('Performance Scores by Document')
ax.set_ylim(0, 1)
plt.xticks(rotation=90)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Score')

plt.tight_layout()
plt.show()