import random

class die:
    
    def __init__(self, hunger=False) -> None:
        self.sample_space = list(range(1,11))    
        self.sample_space[0] = 'b' if hunger else 0
    
        
    
        
d = die(hunger=True)

print(d.sample_space)