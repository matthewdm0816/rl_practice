from http.client import EXPECTATION_FAILED
from utils import *
import copy
from matplotlib import pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, wait
H, W = 4, 12
STEPS = 1_000
TOTAL_EXPERIMENTS = 100
RW_CLASSES = [
    CliffStatesTD0ExpectedSarsa,

    CliffStatesv2,
    CliffStatesEpsGreedy,
    CliffStatesOffPolicyPolicyIterationEpsGreedy,
    CliffStatesTD0Sarsa,
]
NAMES = [
    "On-policy MCES",
    "On-policy MC $\\varepsilon$-greedy",
    "Off-policy MC Policy Iteration",
    "TD(0) Sarsa",
    "TD(0) Expected Sarsa",
]
rr_records = {}
window_size = STEPS // 20

if __name__ == "__main__":
    for name, rw_class in zip(NAMES, RW_CLASSES):
        print(f"Calculating {name}")
        all_rrs = []
        # for _ in range(TOTAL_EXPERIMENTS):
            # states = rw_class(H, W)
            # states.init_values()
            # round_rewards = []
            # for step in range(STEPS):
            #     rw = states.random_walk()
            #     round_rewards.append(rw)

            # states.show_policy()
            # print("-" * 20)
            # all_rrs.append(round_rewards)
        executor = ProcessPoolExecutor(max_workers=16)
        all_rrs = executor.map(run_exp, [(rw_class, H, W, STEPS) for _ in range(TOTAL_EXPERIMENTS)])
        all_rrs = list(all_rrs)
        all_rrs = np.array(all_rrs)
        # all_rrs = window_average_ndim(all_rrs, window_size=STEPS//100)
        rr_mean, rr_std = compute_mean_std(all_rrs)

        rr_mean = window_average(rr_mean, window_size=window_size)
        # rr_mean = smoothed(rr_mean, gamma=0.8)
        # rr_std = rr_std[:-window_size+1]
        rr_std = window_average(rr_std, window_size=window_size) / (
            TOTAL_EXPERIMENTS**0.5
        )
        rr_records[name] = (rr_mean, rr_std)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    clrs = sns.color_palette("husl", 5)
    with sns.axes_style("darkgrid"):
        for idx, (rr_mean, rr_std) in enumerate(rr_records.values()):
            ax.plot(rr_mean, c=clrs[idx])
            ax.fill_between(
                np.arange(rr_mean.shape[0]),
                rr_mean + rr_std,
                rr_mean - rr_std,
                alpha=0.3,
                facecolor=clrs[idx],
            )
    legends = []
    for name in rr_records.keys():
        legends.append(f"{name}, mean over {TOTAL_EXPERIMENTS} times")
        # legends.append(f"{name}, 0.1 std")
        legends.append("_nolegend_")
    ax.legend(legends, loc="lower right")
    ax.set_xlabel("Round")
    ax.set_ylabel("Round Reward")
    plt.ylim(-40,-20)

    fig.savefig("mc.png", dpi=500)
