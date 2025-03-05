from vocabularyloader import VocabularyLoader
from memoryhandler import MemoryHandler
from datetime import datetime
import sys
import matplotlib.pyplot as plt


sys.stdout.reconfigure(encoding='utf-8')
course_name = 'french'
vocab = VocabularyLoader(course_name).load()
memory = MemoryHandler(vocab,course_name).load()
memory_data_copy = memory.get_data_copy()
ones = []
max_gap = 0
max_gap2 = 0
up = 0
down = 0

for entry in memory_data_copy.values():
    for i in range(len(entry["time"])):
        ones.append((entry["time"][i],entry["values"][i]))
    if len(entry["time"]) > 1:
        gap = entry["time"][-1]-entry["time"][-2]
        if gap > max_gap:
            max_gap = gap
    if len(entry["time"]) > 0:
        gap = int(datetime.now().timestamp())-entry["time"][-1]
        if gap > max_gap2:
            max_gap2 = gap
        
        if entry["values"][-1] == 1:
            up += 1
            down += 1
        else:
            down += 1

print(max_gap/86400, max_gap2/86400)
print(len(memory_data_copy.values()))
print(len(ones))
print(up)
print(down)
print(up/down)

ones.sort(key=lambda x: x[0])

sum = 0
for i in range(len(ones)):
    sum += ones[i][1]
    ones[i] = (i+1,sum/(i+1))

x, y = zip(*ones)

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')

plt.grid(True)
plt.show()