import math

class TrainSet:
    def __init__(self,data):
        self.data = data
    
    def convert(self):
        result = []
        entries_count = 0

        for entry in self.data.values():
            inputs = []
            outputs = []
            time = entry["time"]
            values = entry["values"]
            
            for i in range(len(time)-1):
                inputs.append(((values[i]+1) % 2, values[i] % 2, math.log(math.e+time[i+1]-time[i])))
                outputs.append((values[i+1]))
            
            if len(inputs) > 0:
                entries_count += len(inputs)
                result.append({'input' : inputs, 'output' : outputs})
        
        return result, entries_count