from vocabularyloader import VocabularyLoader
from memoryhandler import MemoryHandler
from evalset import EvalSet
from trainset import TrainSet
from showingorder import ShowingOrder
from lstmnetwork import LSTMNetwork
from modeltrainer import ModelTrainer
from datetime import datetime
from pynput import keyboard
import random
import torch
import os
import sys
import config
import atexit


# poznać bibliotekę do GUI
# zaimplementować GUI
# uobiektowić główną pętlę
# zrobić pakiety
# rozważyć zrobienie hermetyzacji zmiennych _ __
# atomizacja zapisywania, żeby nie można było zamknąć w trakcie zapisywania do memory


def get_key():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    return pressed_key

pressed_key = None

def on_press(key):
    global pressed_key
    try:
        pressed_key = key.char
    except AttributeError:
        pressed_key = key
    return False

def on_exit():
    print("Exiting and saving...")
    memory.save()
    print("Saved")

atexit.register(on_exit)

def activate_model_on_data(data, model):
    model.eval()
    
    tensor_seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    length = torch.tensor([len(data)], dtype=torch.int64)
    
    with torch.no_grad():
        output = model(tensor_seq, length)
    
    last_output = output[0, length.item() - 1]
    return last_output.item()

sys.stdout.reconfigure(encoding='utf-8')
course_name = sys.argv[1]
vocab = VocabularyLoader(course_name).load()
memory = MemoryHandler(vocab,course_name).load()
memory_data_copy = memory.get_data_copy()
train, entries_count = TrainSet(memory_data_copy).convert()

model_path = "models/" + course_name + ".pt"
model = LSTMNetwork()

if memory.update_level(entries_count):
    model = ModelTrainer(model, config.epochs, train).train()
    model.save(model_path)

order = ShowingOrder(memory_data_copy).generate()
model = model.load(model_path)


for entry in order:
    question = vocab[entry[0]][entry[1]%2]
    answer = vocab[entry[0]][(entry[1]+1)%2]

    memory_cell = memory.get((entry[0],entry[1]))
    timestamp = int(datetime.now().timestamp())

    previous_gap = (config.min_first_gap/config.max_gap_multiplier)*config.max_gap_multiplier**(len(memory_cell["time"])-1)
    if len(memory_cell["time"]) > 1 and memory_cell["time"][-1]-memory_cell["time"][-2] < previous_gap:
        previous_gap = memory_cell["time"][-1]-memory_cell["time"][-2]

    if len(memory_cell["time"]) > 0:
        new_timestamp = memory_cell["time"][-1]+min(int(previous_gap*config.max_gap_multiplier),config.max_gap)
    else:
        new_timestamp = 0
            
    if new_timestamp > timestamp:
        eval = EvalSet(memory_cell).convert()
        result = activate_model_on_data(eval, model)

        if result > config.recall:
            continue

        print("Likelihood: ", result)
    
    print(question)
    
    while(True):
        a = get_key()
        if a == keyboard.Key.enter:
            break
    
    print(answer)

    while(True):
        recall = get_key()

        if recall in ('2','3'):
            if recall == '2':
                recall = 0
            else:
                recall = 1
            
            print(recall)

            memory.add_entry(entry[0],entry[1],timestamp,recall)
            
            if random.randint(1,config.rep_per_save) == 1:
                print("Saving...")
                memory.save()
            break
    
    os.system("cls")
        

memory.save()