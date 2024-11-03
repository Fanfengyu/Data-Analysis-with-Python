import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Определение архитектуры сверточной нейросети
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Определяем слои сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(5 * 5 * 5, 100)  # вычисляем размер flatten слоя
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 5 * 5 * 5)  # Преобразуем в плоский вид (flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Загрузка и предобработка CIFAR10 датасета
transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root="./cifar10_data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="./cifar10_data", train=False, download=True, transform=transform)

# Разделение тренировочных данных на тренировочную и валидационную выборки
train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# Создание загрузчиков данных
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация модели и перемещение на устройство
model = ConvNet().to(device)

# Определение функции потерь и оптимизатора
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Функция для оценки качества на валидационной или тестовой выборке
def evaluate(model, dataloader, loss_fn):
    model.eval()
    losses = []
    num_correct = 0
    num_elements = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            losses.append(loss.item())

            y_pred = torch.argmax(logits, dim=1)
            num_correct += torch.sum(y_pred == y_batch)
            num_elements += y_batch.size(0)

    accuracy = num_correct.float() / num_elements
    return accuracy.item(), np.mean(losses)

# Функция для тренировки модели
def train(model, loss_fn, optimizer, train_loader, val_loader, n_epoch=10):
    for epoch in range(n_epoch):
        print(f"Эпоха {epoch+1}/{n_epoch}")
        model.train()
        running_losses = []
        running_accuracies = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_losses.append(loss.item())
            y_pred = torch.argmax(logits, dim=1)
            train_accuracy = torch.sum(y_pred == y_batch) / y_batch.size(0)
            running_accuracies.append(train_accuracy.item())

        avg_train_loss = np.mean(running_losses)
        avg_train_acc = np.mean(running_accuracies)
        print(f"Средний лосс на обучении: {avg_train_loss:.4f}, Точность на обучении: {avg_train_acc:.4f}")

        # Оценка на валидационной выборке
        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn)
        print(f"Лосс на валидации: {val_loss:.4f}, Точность на валидации: {val_accuracy:.4f}")

    return model

# Запуск тренировки модели
model = train(model, loss_fn, optimizer, train_loader, val_loader, n_epoch=10)

# Оценка на тестовой выборке
test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print(f"Точность на тесте: {test_accuracy:.4f}")

# Проверка, прошел ли порог точности для сдачи задания
if test_accuracy < 0.5:
    print("Качество на тесте ниже 0.5, 0 баллов")
elif test_accuracy < 0.6:
    print("Качество на тесте между 0.5 и 0.6, 0.5 баллов")
else:
    print("Качество на тесте выше 0.6, 1 балл")

# Сохранение модели для сдачи
torch.save(model.state_dict(), "model.pth")
