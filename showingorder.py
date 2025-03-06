from datetime import datetime
import random
import config

class ShowingOrder:
    def __init__(self,data):
        self.data = data
        self.time = datetime.now().timestamp()
    
    def generate(self):
        repetitions = []
        learning = []

        for key, entry in self.data.items():
            if len(entry["time"]) == 0:
                learning.append(key)
            else:
                repetitions.append(key)
        
        repetitions *= config.max_item_rep_in_session
        learning *= config.max_item_rep_in_session
        
        random.shuffle(repetitions)

        random.shuffle(learning)

        return repetitions + learning