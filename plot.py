import matplotlib.pyplot as plt
import numpy as np

labels = [
    'ARXIV_1304_4762',
    'ARXIV_2101_12333',
    'ARXIV_2206_06633',
    'ARXIV_astro_ph_0205031',
    'ARXIV_physics_9806027'
]
scores = np.array([0.8571428571428571, 0.8, 1.0, 0.375, 0.75])
mean_score = scores.mean()

cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(scores)))

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, scores, color=colors)

ax.axhline(mean_score, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_score:.2f}')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylim(0, 1.1)
ax.set_title('Tagging Accuracy', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=12)

ax.legend()

plt.tight_layout()
plt.show()