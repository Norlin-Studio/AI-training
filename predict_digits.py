from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 以這支程式所在的資料夾尋找模型
folder = Path(__file__).resolve().parent

# 1. 建立與訓練時相同的模型結構
model = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 2. 載入之前學到的參數
parameters = torch.load(
    folder / "digit_model.pth",
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(parameters)
model.eval()

# 3. 讀取測試圖片
test_data = datasets.MNIST(
    root=str(folder / "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# 4. 顯示前 12 張圖片和預測
fig, axes = plt.subplots(3, 4, figsize=(10, 7))

with torch.no_grad():
    for index, ax in enumerate(axes.flat):
        image, answer = test_data[index]

        scores = model(image.unsqueeze(0))
        prediction = scores.argmax(dim=1).item()

        color = "green" if prediction == answer else "red"

        ax.imshow(image.squeeze(0), cmap="gray")
        ax.set_title(
            f"Guess: {prediction} / Answer: {answer}",
            color=color
        )
        ax.axis("off")

plt.tight_layout()
plt.show()