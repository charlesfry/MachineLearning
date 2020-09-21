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
        sample = np.random.normal(loc=mean,scale=std,size=(1,1))[0][0]

        # limit increase/decrease to 5 * standard deviation
        sample = max(sample, mean - 5*std)
        sample = min(sample, mean + 5*std)
        return int(np.round(sample))

    def create_monster(self,seed):
        self.seed_everything(seed)
        name = self.name
        type1 = self.type1
        type2 = self.type2
        hp = self.draw_sample(*self.health)
        str = self.draw_sample(*self.str)
        dfn = self.draw_sample(*self.dfn)
        dex = self.draw_sample(*self.dex)
        int = self.draw_sample(*self.int)
        agil = self.draw_sample(*self.agil)
        chr = self.draw_sample(*self.chr)

        return Monster(name,type1,type2,hp,str,dfn,dex,int,agil,chr)



if __name__ == '__main__' :
    cr = Monster_Creator('Fuffmup','Woodlander','None',(100,5),(10,2),(6,1),(14,1.5),(11,.78),(16,3),(19,1))
    mon = cr.create_monster(seed=0)
    print(vars(mon))