import numpy as np
import random
import os

class Monster :
    def __init__(self, name, type1, type2, health, str, dfn, dex, int, agil, chr):
        self.name = name
        self.type1 = type1
        self.type2 = type2
        self.health = health
        self.str = str
        self.dfn = dfn
        self.dex = dex
        self.int = int
        self.agil = agil
        self.chr = chr

    @staticmethod
    def get_nickname():
        return 'pog champ'

    @staticmethod
    def seed_everything(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

class Monster_Creator(Monster) :
    def __init__(self, name, type1, type2, health, str, dfn, dex, int, agil, chr):
        super().__init__(name, type1, type2, health, str, dfn, dex, int, agil, chr)

    @staticmethod
    def draw_sample(mean,std):
        return np.random.normal(loc=mean,scale=std,size=(1,1))

    def create_monster(self,seed):
        self.seed_everything(seed)
        name = self.name
        type1 = self.type1
        type2 = self.type2






if __name__ == '__main__' :
    mon = Monster_Creator('Fuffmup','Woodlander','None',(100,5),(10,2),(6,1),(14,1.5),(11,.78),(16,3),(19,1))
    butt = mon.create_monster()
