import torch
from torch import nn
from torchvision import datasets, transforms

# 1. 讀取資料，並把圖片轉成 0～1 的數值
data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

image, answer = data[0]

# 2. 建立神經網路
model = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 3. 加上一個「圖片張數」的維度
# 從 [1, 28, 28] 變成 [1, 1, 28, 28]
inputs = image.unsqueeze(0)

# 4. 讓模型猜一次
with torch.no_grad():
    scores = model(inputs)

prediction = scores.argmax(dim=1).item()

print("模型對數字 0～9 的分數：")
print(scores[0].tolist())
print("模型猜測：", prediction)
print("正確答案：", answer)

# 評分方式：衡量分類預測的錯誤
loss_function = nn.CrossEntropyLoss()

# 更新規則：梯度下降，學習率為 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 正確答案，格式是一批圖片的答案
target = torch.tensor([answer], dtype=torch.long)

model.train()

for step in range(1, 51):
    # 1. 預測
    scores = model(inputs)

    # 2. 計算損失
    loss = loss_function(scores, target)

    # 3. 清除上一次的梯度
    optimizer.zero_grad()

    # 4. 計算這一次的梯度
    loss.backward()

    # 5. 根據梯度更新參數
    optimizer.step()

    if step == 1 or step % 10 == 0:
        prediction = scores.argmax(dim=1).item()
        print(
            f"第 {step:2d} 次："
            f"損失 = {loss.item():.4f}，"
            f"更新前的猜測 = {prediction}"
        )

# 檢查最後一次更新後的預測
with torch.no_grad():
    final_prediction = model(inputs).argmax(dim=1).item()

print("訓練後猜測：", final_prediction)
print("正確答案：", answer)