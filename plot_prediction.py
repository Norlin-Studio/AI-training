import matplotlib.pyplot as plt

# 真實資料
volumes = [300, 500, 700]
prices = [30, 50, 70]

# 模型：預測價格 = 容量 × weight
def predict(volume, weight):
    return volume * weight

# 畫線用的容量
x = [0, 100, 200, 300, 400, 500, 600, 700, 800]

# 比較兩組參數
before = [predict(v, 0.15) for v in x]
after = [predict(v, 0.1) for v in x]

plt.plot(x, before, label="Before: weight = 0.15",
         linestyle="--", color="red")
plt.plot(x, after, label="After: weight = 0.10",
         color="blue")

# 真實資料用圓點表示
plt.scatter(volumes, prices, color="black",
            label="Actual data", zorder=3)

plt.xlabel("Volume (ml)")
plt.ylabel("Price")
plt.title("Drink price prediction")
plt.legend()
plt.grid(True)
plt.show()