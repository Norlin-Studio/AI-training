import matplotlib.pyplot as plt

data = [(300, 30), (500, 50), (700, 70)]

# 給定一個 weight，計算模型錯得多嚴重
def loss(weight):
    return sum(
        (volume * weight - price) ** 2
        for volume, price in data
    ) / len(data)

# 試算從 0 到 0.2 的各種 weight
weights = [i / 1000 for i in range(201)]
losses = [loss(w) for w in weights]

plt.plot(weights, losses)

# 標出起點與最佳參數
plt.scatter([0.15], [loss(0.15)], color="red",
            label="Start: weight = 0.15")
plt.scatter([0.1], [loss(0.1)], color="green",
            label="Best: weight = 0.1")

plt.xlabel("weight")
plt.ylabel("Mean squared error")
plt.title("Drink model loss")
plt.legend()
plt.grid(True)
plt.show()