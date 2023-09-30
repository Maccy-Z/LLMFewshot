from matplotlib import pyplot as plt
import numpy as np

ys = \
[-0.0, -0.10467866063117981, -0.21554212272167206, -0.33473965525627136, -0.46423810720443726, -0.6094486713409424, -0.7760887742042542]
xs = \
    ['Inc', 'Fed', 'Not-Inc', 'Priv', 'Local', 'State', '?', ]

xticks = range(len(xs))# np.arange(20, 101, 20)
fig, ax1 = plt.subplots(figsize=(4, 4))
ax1.set_xticks(xticks, xs, fontsize=10)

ax1.plot(xs, ys)
ax1.scatter(xs, ys)

yticks = np.arange(0, -0.81, -0.2)
yticks = [round(y, 1) for y in yticks]
ax1.set_yticks(yticks, labels=yticks, fontsize=12)

ax1.set_xlabel("Employment Type", fontsize=13)
ax1.set_ylabel("Activation magnitude", fontsize=13)

"""Comment out to only plot a single axis"""
true_ys = [0.557, 0.386,  0.285, 0.219, 0.295, 0.272, 0.104]
yticks = np.arange(0, 0.8, 0.2)
yticks = [round(y, 1) for y in yticks]

# Make second plot using same x axis
ax2 = ax1.twinx()
ax2.plot(xs, true_ys, color="tab:orange")
ax2.scatter(xs, true_ys, color="tab:orange")
ax2.set_yticks(yticks, labels=yticks, fontsize=12)
ax2.set_ylabel("Marginal probability", fontsize=13)


plt.tight_layout()
#plt.subplots_adjust(left=0.2, right=0.98, top=0.94, bottom=0.15)
plt.title("Income")
plt.show()
