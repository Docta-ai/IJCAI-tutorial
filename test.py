import numpy as np


np.random.seed(0)
num_samples = 1000
T = [
    [0.6, 0.4],
    [0.2, 0.8],
]
p = [0.4, 0.6]
clean_labels = [0] * int(num_samples * p[0]) + [1] * (num_samples - int(num_samples * p[0]))
np.random.shuffle(clean_labels)
noisy_labels = []

for i in clean_labels:
    noisy_labels.append(np.random.choice([0, 1], size = 3, p=T[i]))

# Get true T
trueT = np.zeros((2,2))
for i in range(len(clean_labels)):
    for j in range(len(noisy_labels[0])):
        trueT[clean_labels[i]][noisy_labels[i][j]] += 1
print(trueT / np.sum(trueT, 1).reshape(-1,1))


mv_T = np.zeros((2,2))
from collections import Counter
mv_p = np.zeros(2)
for i in range(len(clean_labels)):
    mv_label = Counter(noisy_labels[i]).most_common(1)[0][0]
    mv_p[mv_label] += 1
    for j in range(len(noisy_labels[0])):
        mv_T[mv_label][noisy_labels[i][j]] += 1
print(mv_T / np.sum(mv_T, 1).reshape(-1,1))
print(mv_p / np.sum(mv_p))


from docta.apis import Diagnose
from docta.core.report import Report
from docta.utils.config import Config
from docta.datasets import CustomizedDataset

cfg = Config.fromfile('./config/toy.py')
report = Report()

dataset = CustomizedDataset(noisy_labels, np.asarray(noisy_labels).reshape(-1))
dataset.consensus_patterns = noisy_labels
# diagnose labels
estimator = Diagnose(cfg, dataset, report = report)
estimator.hoc()
print(report.diagnose)
