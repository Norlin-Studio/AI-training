import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 準備資料
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# 訓練資料打亂順序，每批取 64 張
train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=256, shuffle=False
)

# 2. 建立模型
model = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. 先練習 3 個 epoch
for epoch in range(1, 51):
    model.train()
    total_loss = 0.0

    for images, answers in train_loader:
        scores = model(images)
        loss = loss_function(scores, answers)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(train_data)

    # 4. 用沒參與訓練的圖片考試
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, answers in test_loader:
            scores = model(images)
            predictions = scores.argmax(dim=1)
            correct += (predictions == answers).sum().item()

    accuracy = correct / len(test_data)

    print(
        f"Epoch {epoch}/50："
        f"訓練損失 = {average_loss:.4f}，"
        f"測試正確率 = {accuracy:.2%}",
        flush=True
    )

# 5. 保存訓練成果
torch.save(model.state_dict(), "digit_model.pth")
print("模型已儲存為 digit_model.pth")