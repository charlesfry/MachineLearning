import numpy as np
import pandas as pd
import random
import os
import sys

sys.path.append('../../../GitHub')
from GitHub.MachineLearning.MonsterRancherApp.MonsterClass import Monster_Creatorv

class MonsterList :
    def __init__(self):
        self.monster_list = self.build_list()

    @staticmethod
    def seed_everything(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def build_list(self):
        columns = ['name','type1','type2','health','str','dfn','dex','int','rflx','charisma']
        """
        name: monster's name
        type1: primary monster type. Cannot be None
        type2: secondary monster type. Can be None
                next are the stats. These are shown as tuple descriptions of normal distributions (mu,stdev)
        health: total hit points
        str: strength score. improves physical feats of strength, including physical attacks
        dfn: defense. mitigates total hp damage
        dex: dexterity. Increases odds of hitting and improves ranged attacks
        int: intelligence. Increases mental abilities
        agil: agility. increases dodge chance and critical hit chance
        chr: increases team performance and child stat chances
        """

        all_monsters = [
            Monster_Creator('Fuffmup','Rodent','None',(100,5),(10,2),(6,1),(14,1.5),(11,.78),(16,3),(19,1)),
            Monster_Creator('Dib','Lizard','Snake',(80,5),(12,2),(8,1),(22,3),(18,2),(18,1.5),(12,4)),
            Monster_Creator('Valer','Voidcreep', 'MindBender', (80,2), (6, 1), (11, 2.2), (10, 1.5), (22,3), (19,3), (15,3))
        ]

        monster_list = pd.DataFrame(data=all_monsters,columns=columns)

        return monster_list

    def get_list_(self):
        return self.monster_list

    @staticmethod
    def get_stats(row:list):
        output = []
        for entry in row :
            if type(entry) == str :
                output.append(entry)
                continue


    def get_barcode_monster(self,seed):
        self.seed_everything(seed)
        monster_list = self.monster_list.copy()
        row = monster_list.sample(n=1)

        monster =0


        return row



    def get_child_monster(self,father,mother):
        pass

if __name__ == '__main__' :
    monsters = MonsterList()
    selected_monster = monsters.get_barcode_monster(seed=69)
    print(selected_monster)
