import numpy as np
import matplotlib.pyplot as plt

EPSILONS = [0, 0.1, 0.01]
RUNS = 2000
TIME = 1000


class Bandit:
    def __init__(self, EPSILON = 0, SAMPLE_AVERAGES = False, K_ARM = 10, TRUE_REWARD = 0., INITIAL = 0, STEP_SIZE = 0.01):
        self.EPSILONS = EPSILON
        self.TIME = 0
        self.STEP_SIZE = STEP_SIZE
        self.K = K_ARM
        self.INITIAL = INITIAL
        self.SAMPLE_AVERAGES = SAMPLE_AVERAGES
        self.TRUE_REWARD = TRUE_REWARD
        self.INDICES = np.arange(self.K)


    def reset(self):
        self.Q_TRUE = np.random.rand(self.K)
        self.Q_ESTIMATE = np.zeros(self.K)
        self.ACTION_COUNT = np.zeros(self.K)
        self.BEST_ACTION = np.argmax(self.Q_TRUE)
        self.TIME = 0

    def act(self):
        if np.random.rand() < self.EPSILONS:
            return np.random.choice(self.INDICES)

        Q_BEST = np.max(self.Q_ESTIMATE)
        return np.random.choice(np.where(self.Q_ESTIMATE == Q_BEST)[0])

    def step(self, ACTION):
        REWARD = np.random.rand() + self.Q_TRUE[ACTION]
        self.TIME += 1
        self.ACTION_COUNT[ACTION] += 1

        if self.SAMPLE_AVERAGES:
            self.Q_ESTIMATE[ACTION] += (REWARD - self.Q_ESTIMATE[ACTION]) / self.ACTION_COUNT[ACTION]
        else:
            self.Q_ESTIMATE[ACTION] += self.STEP_SIZE * (REWARD - self.Q_ESTIMATE[ACTION])

        return REWARD
            
        
def simulate(RUNS, TIME, BANDITS):
    REWARDS = np.zeros((len(BANDITS), RUNS, TIME))
    BEST_ACTION_COUNTS = np.zeros(REWARDS.shape)

    for i, BANDIT in enumerate(BANDITS):
        for r in range(RUNS):
            BANDIT.reset()
            for t in range(TIME):
                ACTION = BANDIT.act()
                REWARD = BANDIT.step(ACTION)
                REWARDS[i, r, t] = REWARD
                if ACTION == BANDIT.BEST_ACTION:
                    BEST_ACTION_COUNTS[i, r, t] = 1

    MEAN_BEST_ACTION_COUNTS = BEST_ACTION_COUNTS.mean(axis = 1)
    MEAN_REWARDS = REWARDS.mean(axis = 1)
    return MEAN_BEST_ACTION_COUNTS, MEAN_REWARDS



BANDITS = [Bandit(EPSILON = EPS, SAMPLE_AVERAGES = True) for EPS in EPSILONS]
BEST_ACTION_COUNTS, REWARDS = simulate(RUNS, TIME, BANDITS)
plt.figure(figsize = (10, 20))

plt.subplot(2, 1, 1)
for EPS, REWARDS in zip(EPSILONS, REWARDS):
    plt.plot(REWARDS, label = 'EPSILON = %.02f' % (EPS))
plt.xlabel('STEPS')
plt.ylabel('AVERAGE REWARD')
plt.legend()

plt.subplot(2, 1, 2)
for EPS, COUNTS in zip(EPSILONS, BEST_ACTION_COUNTS):
    plt.plot(COUNTS, label = 'EPSILON = %.02f' % (EPS))
plt.xlabel('STEPS')
plt.ylabel('% OPTIMAL ACTION')
plt.legend()

plt.show()