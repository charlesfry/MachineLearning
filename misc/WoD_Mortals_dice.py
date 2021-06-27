import math
import random

class Die:
    def __init__(self, success_threshold=8, sides=10, crit_threshold=math.inf,
                 repeat_crit=None, sample_space=None, crit_bonus=None):
        self.success_threshold = success_threshold
        self.sides = sides
        self.crit_threshold = crit_threshold
        self.repeat_crit = repeat_crit
        self.sample_space = sample_space if sample_space is not None else list(range(1, sides + 1))
        self.crit_bonus = crit_bonus

    def roll(self, return_values=False):
        result = random.choice(self.sample_space)
        is_crit = result >= self.crit_threshold

        if is_crit and self.crit_bonus:
            result += self.crit_bonus

        if return_values: return result


        if is_crit and self.repeat_crit:
            return 1 + self.roll()

        return 1 if result >= self.success_threshold else 0

import numpy as np
from scipy import stats

def evaluate_successes(iters, dice_per_iter=1):
    results = np.empty(iters)

    dice = [Die(crit_threshold=10, repeat_crit=True) for _ in range(dice_per_iter)]

    for i in range(iters):
        successes = 0
        successes += sum(j.roll() for j in dice)
        results[i] = successes

    return results

samples = np.array([evaluate_successes(1000).mean() for _ in range(30)])

def calculate_p_value(samples:np.array):
    mean = samples.mean()
    std = samples.std()
    mu = 1/3

    z_score = (mean - mu) / std
    return stats.norm.sf(z_score)

print(calculate_p_value(samples))