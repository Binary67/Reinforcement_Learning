import matplotlib.pyplot as plt
import numpy as np

RUNS = 2000
TIME = 1000

# Class
class Bandit:
    def __init__(self, EPSILON = 0, INITIAL = 0, STEP_SIZE = 0, K_ARM = 10):
        self.EPSILON = EPSILON
        self.INITIAL = INITIAL
        self.STEP_SIZE = STEP_SIZE
        self.K_ARM = K_ARM
        self.TIME = 0
        self.INDICES = np.arange(self.K_ARM)

    def reset(self):
        self.Q_TRUE = np.random.randn(self.K_ARM)
        self.Q_ESTIMATE = np.zeros(self.K_ARM) + self.INITIAL
        self.ACTION_COUNT = np.zeros(self.K_ARM)
        self.Q_BEST_TRUE = np.argmax(self.Q_TRUE)
        self.TIME = 0

    def act(self):
        if np.random.rand() < self.EPSILON:
            return np.random.choice(self.INDICES)
        
        Q_BEST_ESTIMATE = np.max(self.Q_ESTIMATE)

        return np.random.choice(np.where(self.Q_ESTIMATE == Q_BEST_ESTIMATE)[0])

    def step(self, ACTION):
        REWARD = np.random.randn() + self.Q_TRUE[ACTION]
        self.TIME += 1
        self.ACTION_COUNT[ACTION] += 1

        self.Q_ESTIMATE[ACTION] += self.STEP_SIZE * (REWARD - self.Q_ESTIMATE[ACTION])

        return REWARD
    

def simulate(RUNS, TIMES, LIST_BANDITS):
    REWARDS = np.zeros((len(LIST_BANDITS), RUNS, TIMES))
    BEST_ACTION_COUNT = np.zeros(REWARDS.shape)

    for i, BANDIT in enumerate(LIST_BANDITS):
        for r in range(RUNS):
            BANDIT.reset()
            for t in range(TIMES):
                ACTION = BANDIT.act()
                REWARD = BANDIT.step(ACTION)
                REWARDS[i, r, t] = REWARD
                if ACTION == BANDIT.Q_BEST_TRUE:
                    BEST_ACTION_COUNT[i, r, t] = 1

    MEAN_BEST_ACTION_COUNT = BEST_ACTION_COUNT.mean(axis = 1)

    return MEAN_BEST_ACTION_COUNT

LIST_BANDITS = []
LIST_BANDITS.append(Bandit(EPSILON = 0, INITIAL = 5, STEP_SIZE = 0.1))
LIST_BANDITS.append(Bandit(EPSILON = 0.1, INITIAL = 0, STEP_SIZE = 0.1))

BEST_ACTION_COUNT = simulate(RUNS, TIME, LIST_BANDITS)

plt.plot(BEST_ACTION_COUNT[0], label = 'EPSILON = 0, INITIAL = 5')
plt.plot(BEST_ACTION_COUNT[1], label = 'EPSILON = 0.1, INITIAL = 0')
plt.xlabel('STEPS')
plt.ylabel('% OPTIMAL ACTION')
plt.legend()

plt.show()