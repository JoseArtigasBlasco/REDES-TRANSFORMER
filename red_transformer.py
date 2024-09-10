# Este código implementa un modelo Transformer para clasificación de texto utilizando PyTorch.

import torch
import torch.nn as nn
import torch.optim as optim

d_model = 512  # Dimensión de la representación vectorial en el modelo.
nhead = 8      # Número de cabezas en el mecanismo de atención multi-cabeza.
num_encoder_layers = 6  # Número de capas en el codificador del Transformer.
num_decoder_layers = 6  #Número de capas en el decodificador del Transformer.
dim_feedforward = 2048  # Dimensión de la capa feedforward en cada capa del Transformer.
dropout = 0.1           # Tasa de dropout para evitar el sobreajuste.
max_seq_length = 100    # Longitud máxima de las secuencias de entrada.
vocab_size = 10000      # Tamaño del vocabulario (número de palabras únicas que el modelo puede manejar).
num_classes = 10        # Número de clases que el modelo debe predecir.

# Definimos modelo Transformer (hereda de nn.Module)
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        src = src.permute(1, 0, 2)  # (lote, secuencia, dimensión) -> (secuencia, lote, dimensión)
        tgt = tgt.permute(1, 0, 2)  # (lote, secuencia, dimensión) -> (secuencia, lote, dimensión)
        memory = self.transformer.encoder(src)
        output = self.transformer.decoder(tgt, memory)
        output = self.fc(output[-1, :, :])  # Tomamos la última salida del decodificador
        return output

def generate_data(batch_size, seq_length):
    src = torch.randint(0, vocab_size, (batch_size, seq_length))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = torch.randint(0, num_classes, (batch_size,))
    return src, tgt, labels

def split_data(data, labels, train_ratio=0.7, val_ratio=0.15):
    total_size = len(labels)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    train_labels = labels[:train_size]
    val_labels = labels[train_size:train_size + val_size]
    test_labels = labels[train_size + val_size:]

    return train_data, val_data, test_data, train_labels, val_labels, test_labels

# Inicialización del modelo, pérdida y optimizador
model = TransformerModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
def train(model, criterion, optimizer, train_data, train_labels, val_data, val_labels, epochs=10, batch_size=32):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(train_labels), batch_size):
            src_batch = train_data[i:i + batch_size]
            tgt_batch = train_data[i:i + batch_size]
            labels_batch = train_labels[i:i + batch_size]

            optimizer.zero_grad()
            output = model(src_batch, tgt_batch)
            loss = criterion(output, labels_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = evaluate(model, criterion, val_data, val_labels, batch_size)
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {epoch_loss / len(train_labels):.4f}, Validation Loss: {val_loss / len(val_labels):.4f}')

# Evaluación del modelo
def evaluate(model, criterion, data, labels, batch_size=32):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            src_batch = data[i:i + batch_size]
            tgt_batch = data[i:i + batch_size]
            labels_batch = labels[i:i + batch_size]

            output = model(src_batch, tgt_batch)
            loss += criterion(output, labels_batch).item()
    return loss

# Probamos el modelo
def test(model, data, labels, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            src_batch = data[i:i+batch_size]
            tgt_batch = data[i:i+batch_size]
            labels_batch = labels[i:i+batch_size]

            output = model(src_batch, tgt_batch)
            _, predicted = torch.max(output.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

# Dividimos datos en entrenamiento, validación y prueba
data, tgt, labels = generate_data(1000, max_seq_length)
train_data, val_data, test_data, train_labels, val_labels, test_labels = split_data(data, labels)

# Entrenar el modelo
train(model, criterion, optimizer, train_data, train_labels, val_data, val_labels)

# Probar el modelo
test(model, test_data, test_labels)

