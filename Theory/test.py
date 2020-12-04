import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from typing import Union, Optional, Dict

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 69
seed_everything(seed=seed)

class Node:
    def __init__(self, reward:Union[int, float], children:Optional[Dict[str, float]]=None):
        self.terminal = children is None
        self.children = children
        self.next_node = self.transition()

    def transition(self):
        output_node = None
        if self.terminal: return output_node
        roll = random.random()
        total = 0
        for node in self.children.keys():
            output_node = node
            total += self.children[node]
            if total >= roll: return output_node
        return output_node


class CavesNode(Node):
    """
    this is the class of nodes for the caves challenge for CMU 10-701 HW 1
    """

    def __init__(self, reward: Union[int, float], children: Optional[Dict[str, float]] = None):
        super().__init__(reward=reward, children=children)
