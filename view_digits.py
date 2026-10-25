import matplotlib.pyplot as plt
from torchvision import datasets

# 第一次執行會下載資料，之後直接讀取
data = datasets.MNIST(
    root="data",
    train=True,
    download=True
)

# 取出第一張圖片，以及它的正確答案
image, answer = data[0]

print("訓練圖片數量：", len(data))
print("圖片大小：", image.size)
print("這張圖片的答案：", answer)

# 顯示圖片
plt.imshow(image, cmap="gray")
plt.title(f"Correct answer: {answer}")
plt.axis("off")
plt.show()