import torch
import torch.nn.utils.rnn as rnn_utils
from lstmnetwork import LSTMNetwork

# Dane wejściowe
data = [
        [(0, 1, 4.63199011305389), (1, 0, 5.712802610069324), (0, 1, 7.82513273289133), (0, 1, 9.798278041133722)],
        [(0, 1, 4.63199011305389), (1, 0, 5.712802610069324), (0, 1, 7.82513273289133), (0, 1, 10.798182590890704)],
        ]

# Funkcja do aktywacji sieci
def activate_model_on_data(data, model):
    model.eval()

    inputs = []
    lengths = []

    # Przygotowanie danych: tensorów wejściowych i ich długości
    for seq in data:
        tensor_seq = torch.tensor(seq, dtype=torch.float32)  # Tworzenie tensora
        inputs.append(tensor_seq)
        lengths.append(len(seq))

    # Padding sekwencji
    padded_inputs = rnn_utils.pad_sequence(inputs, batch_first=True, padding_value=0.0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

    # Inferencja modelu
    with torch.no_grad():
        outputs = model(padded_inputs, lengths_tensor)

    # Pobranie wyników dla ostatnich neuronów
    batch_results = []
    for i, length in enumerate(lengths):
        last_output = outputs[i, length - 1]  # Ostatni neuron dla odpowiedniej długości
        batch_results.append(last_output.item())

    return batch_results



# Inicjalizacja modelu
model = LSTMNetwork()

# Aktywacja modelu na danych
results = activate_model_on_data(data, model)

# Wyświetlenie wyników
print(results)