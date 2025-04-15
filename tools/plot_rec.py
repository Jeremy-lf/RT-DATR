from matplotlib import pyplot as plt
import matplotlib.patches as mptch
import numpy as np
from math import pi

# Sample data
categories = ['cityscapes->foggycityscapes','cityscapes->bdd100k', 'sim10k->cityscapes', 'kitti->cityscapes']
n_categories = len(categories)

# Example data for each model
values = [
    [52.7, 40.7, 67.2, 50.3],
    [51.2, 33.7, 62.0, 0.0],
    [47.1, 29.4, 53.4, 0.0],
    [46.8, 0.0, 0.0, 0.0],
    [41.3,28.9, 52.6, 46.7],
    [43.5, 0,54.7, 48.0]
    # Add more models here
]

# Normalize the values
new_v = []
max_v = [52.7, 40.7, 67.2, 50.3]
for v in values:
    new_v.append([v[i]/max_v[i] for i in range(len(v))])
print(new_v)

models = ['RT-DATR(ours)', 'MRT', 'AQT', 'O2Net', 'DA-DETR', 'SFA']
# Add the first value to the end to close the circle
# for v in new_v:
#     v.append(v[0])




# Create radar chart
angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Hide the outer circle
ax.spines['polar'].set_visible(False)
# Hide the y-axis labels
ax.set_yticklabels([])


# Plot each model
for i, v in enumerate(new_v):
    v.append(v[0])
    print(angles, v)
    ax.plot(angles, v, linewidth=1, linestyle='solid', label=models[i])
    ax.fill(angles, v, alpha=0.1)


# # Add labels with rotation
# for i, angle in enumerate(angles[:-1]):
#     rotation = np.degrees(angle)
#     if angle > pi:
#         rotation += 180
#     ha = 'right' if angle > pi else 'left'
#     print(rotation, ha, categories[i])
#     ax.text(angle, ax.get_ylim()[1] + 0.1, categories[i], size=8, horizontalalignment=ha, verticalalignment='center', rotation=rotation)


print(ax.get_ylim(), ax.get_xlim())
ax.text(angles[2], ax.get_ylim()[1] + 0.065, categories[2], size=10, horizontalalignment='left', verticalalignment='center', rotation=-270)
ax.text(angles[0], ax.get_ylim()[1] + 0.04, categories[0], size=10, horizontalalignment='left', verticalalignment='center', rotation=360+90)

# Add labels
plt.xticks([angles[1], angles[3]], [categories[1], categories[3]], color='black', size=10)

# Add outer circle labels
outer_labels = [52.7, 40.7, 67.2, 50.3]  # Example values
# for i, angle in enumerate(angles[:-1]):
#     ax.text(angle, ax.get_ylim()[1] * 0.98, str(outer_labels[i]), color='black', fontsize=10, fontweight='bold', ha='center', va='center')

ax.text(angles[1], ax.get_ylim()[1] * 0.98+0.01, str(outer_labels[1]),  fontsize=10, ha='center', va='center')
ax.text(angles[3], ax.get_ylim()[1] * 0.98+0.01, str(outer_labels[3]),  fontsize=10,  ha='center', va='center')

ax.text(angles[0]-0.06, ax.get_ylim()[1] * 0.98-0.015, str(outer_labels[0]),  fontsize=10, ha='center', va='center')
ax.text(angles[2]+0.06, ax.get_ylim()[1] * 0.98-0.015, str(outer_labels[2]),  fontsize=10, ha='center', va='center')

# Add legend
# plt.legend(loc='lower right',bbox_to_anchor=(-0.05, -0.05)) 
plt.legend(loc='lower left',bbox_to_anchor=(-0.05, -0.05))

# Show plot
# plt.savefig('radar_chart_3.png',bbox_inches='tight')
plt.savefig('radar_chart_3.pdf', bbox_inches='tight')