import json
import math
import config
from copy import deepcopy

class MemoryHandler:
    def __init__(self,vocab,course_name):
        self.vocab = vocab
        self.course_name = course_name
        self.data = None
    
    def get_data_copy(self):
        temp = deepcopy(self.data)
        del temp[(-1,0)]

        return temp

    def get(self,key):
        return self.data[key]
    
    def load(self):
        try:
            with open("records/" + self.course_name + ".json", "r", encoding="utf-8") as f:
                self.data = {eval(k): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            with open("records/" + self.course_name + ".json", "w", encoding="utf-8") as f:
                json.dump({}, f)
            self.data = {}

        if (-1,0) not in self.data:
            self.data[(-1,0)] = 0
        
        for i in range(len(self.vocab)):
            for poa in (0,1):
                if (i,poa) not in self.data:
                    self.data[(i,poa)] = {
                        "time": [],
                        "values": []
                    }

        self.save()

        return self

    def add_entry(self,line,poa,date,result):
        self.data[(line,poa)]["time"].append(date)
        self.data[(line,poa)]["values"].append(result)

        return self
    
    def update_level(self,entries_count):
        level = int(math.log(entries_count+1e-10)/math.log(config.entries_multiplier))

        if level > self.data[(-1,0)]:
            self.data[(-1,0)] = level

            self.save()

            return True
        
        return False
    
    def save(self):
        with open("records/" + self.course_name + ".json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.data.items()}, f, indent=4, ensure_ascii=False)

        return self