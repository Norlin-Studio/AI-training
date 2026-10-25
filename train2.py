import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 每筆資料是（容量：毫升，價格：元）。這是刻意簡化的教學資料。
data = [(300, 30), (500, 50), (700, 70)]

# 模型：預測價格 = 容量 * weight
# weight 是要學習的參數，代表每毫升多少元。
weight = 0.15
learning_rate = 0.000003  # 因為容量數字較大，這裡使用較小的學習率。
training_steps = 10


def mean_squared_error(current_weight):
    """平均平方誤差：越小表示這批資料的預測越準。"""
    return sum((volume * current_weight - price) ** 2
               for volume, price in data) / len(data)


print("飲料價格模型：預測價格 = 容量 × 參數")
print("訓練資料：300ml → 30元、500ml → 50元、700ml → 70元")
print("\n先猜看看：模型會把參數往上調，還是往下調？")
print("\n次數    參數(元/ml)    平均平方誤差(元²)    500ml預測(元)")
print(f"{0:4d}    {weight:.6f}        {mean_squared_error(weight):12.6f}    {500 * weight:10.3f}")

for step in range(1, training_steps + 1):
    # 1. 預測並與答案比較，計算損失對 weight 的梯度。
    # 平方誤差的微分：2 × 容量 ×（預測價格 − 真實價格）。
    gradient = sum(2 * volume * (volume * weight - price)
                   for volume, price in data) / len(data)

    # 2. 梯度下降：往降低損失的方向更新參數。
    weight = weight - learning_rate * gradient

    # 3. 查看更新後的結果。每輪使用全部三筆資料。
    print(f"{step:4d}    {weight:.6f}        {mean_squared_error(weight):12.6f}    {500 * weight:10.3f}")

print("\n訓練完成！現在用學到的參數預測沒出現在訓練資料裡的容量：")
for new_volume in [400, 600]:
    print(f"{new_volume}ml → 預測 {new_volume * weight:.2f} 元")
