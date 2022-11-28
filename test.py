from sys import platlibdir
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl

a = np.linspace(0, 1, 100) *0.7
b = np.linspace(1, 0, 50) *0.4+0.1
data = np.concatenate((a, b))
# print(data)
colors = sn.color_palette("flare", as_cmap=True)

norm = mpl.colors.Normalize(data.min(), data.max())
sm = cm.ScalarMappable(cmap=colors, norm=norm)
sm.set_array([])

fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(211)
ax.set_frame_on(False)
ax.set_title("title")
# ax.set_title("Yes")
for idx, val in enumerate(data):
    #print(idx, row.value, row.colors)
    if idx == data.shape[0] - 1:
        break
    line = sn.lineplot(x=[idx, idx+1], y=[0,0], color=colors(norm(val)),ax=ax)
    line.tick_params(left=False)
    line.tick_params(bottom=False)
    line.set(xlabel=None)
    line.set(ylabel="Yes")
    line.set(xticklabels=[])
    line.set(yticklabels=[])

data = np.concatenate((b, a))
ax = fig.add_subplot(212)
ax.set_frame_on(False)
# ax.set_title("No")
for idx, val in enumerate(data):
    #print(idx, row.value, row.colors)
    if idx == data.shape[0] - 1:
        break
    line = sn.lineplot(x=[idx, idx+1], y=[0,0], color=colors(norm(val)),ax=ax)
    line.tick_params(left=False)
    line.tick_params(bottom=False)
    line.set(xlabel=None)
    line.set(ylabel="No")
    line.set(xticklabels=[])
    line.set(yticklabels=[])


ax.figure.colorbar(sm, location='bottom', ticks=[data.min(), data.max()],drawedges=False, fraction=0.5, shrink=0.7, aspect=60)
# sn.despine(left=True)
plt.show()
