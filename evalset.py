import math
from copy import deepcopy
from datetime import datetime
import config

class EvalSet:
    def __init__(self,entry):
        self.entry = entry
    
    def convert(self):
        time = deepcopy(self.entry["time"])
        values = deepcopy(self.entry["values"])
        
        t = int(datetime.now().timestamp())-time[-1]
        time.append(t)

        temp = [((values[i]+1) % 2, values[i] % 2, math.log(math.e+time[i+1]-time[i])) for i in range(len(time)-2)]                    
        temp.append(((values[-1]+1) % 2, values[-1] % 2, math.log(math.e+t)))
        
        return temp