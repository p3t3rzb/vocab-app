from datetime import datetime
import random
import config

class ShowingOrder:
    def __init__(self,data):
        self.data = data
        self.time = datetime.now().timestamp()
        self.counter = 0
    
    def generate(self):
        repetitions = []
        learning = []

        for key, entry in self.data.items():
            if entry["scheduled"] == 0:
                learning.append(key)
            else:
                repetitions.append(key)
                if entry["scheduled"] < self.time:
                    self.counter += 1
        
        repetitions *= config.max_item_rep_in_session
        learning *= config.max_item_rep_in_session
        
        random.shuffle(repetitions)

        random.shuffle(learning)
        print(len(repetitions))
        print(len(learning))
        print(self.counter)

        return repetitions + learning