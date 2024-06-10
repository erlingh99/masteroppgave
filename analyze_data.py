import numpy as np
from scipy.stats import chi2
from glob import glob

alpha = 0.05

skip = 5000
length = 10001-skip

CI_NEES_tot = chi2.interval(1-alpha, 6)
CI_ANEES_tot = chi2.interval(1-alpha, 6*100*length, scale=1/(100*length))

CI_NEES_state = chi2.interval(1-alpha, 3)
CI_ANEES_state = chi2.interval(1-alpha, 3*100*length, scale=1/(100*length))



keys = ['se23_body', 'se23_world', 'ESKF_body', 'ESKF_world', 'naive_se23_body', 'naive_se23_world', 'naive_ESKF_body', 'naive_ESKF_world']

for k in keys:
    tot_nees = np.array([])
    vel_nees = np.array([])
    pos_nees = np.array([])

    max_anees = 0
    min_anees = 1e10

    for file in glob("./analysis/sim*.npy"):
        d = np.load(file , allow_pickle=True).item()
        tot_nees = np.concatenate([tot_nees, d[k]["tot"][skip:]])
        vel_nees = np.concatenate([vel_nees, d[k]["vel"][skip:]])
        pos_nees = np.concatenate([pos_nees, d[k]["pos"][skip:]])

        avg = d[k]["tot"][skip:].mean()
        max_anees = max(max_anees, avg)
        min_anees = min(min_anees, avg)

    print(f"\n{k}, total, ANEES stats")
    print(f"Max ANEES {max_anees}")
    print(f"Min ANEES {min_anees}")

    insideCI = (CI_NEES_tot[0] <= tot_nees) * (tot_nees <= CI_NEES_tot[1])
    percents = insideCI.mean()*100
    ANEES = tot_nees.mean()
    print(f"\n{k}, total")
    print(f"Percentage of NEES inside bounds {percents}%")
    print(f"ANEES: {ANEES}, bounds {CI_ANEES_tot}")

    insideCI = (CI_NEES_state[0] <= vel_nees) * (vel_nees <= CI_NEES_state[1])
    percents = insideCI.mean()*100
    ANEES = vel_nees.mean()
    print(f"\n{k}, velocity")
    print(f"Percentage of NEES inside bounds {percents}%")
    print(f"ANEES: {ANEES}, bounds {CI_ANEES_state}")

    insideCI = (CI_NEES_state[0] <= pos_nees) * (pos_nees <= CI_NEES_state[1])
    percents = insideCI.mean()*100
    ANEES = pos_nees.mean()
    print(f"\n{k}, position")
    print(f"Percentage of NEES inside bounds {percents}%")
    print(f"ANEES: {ANEES}, bounds {CI_ANEES_state}")