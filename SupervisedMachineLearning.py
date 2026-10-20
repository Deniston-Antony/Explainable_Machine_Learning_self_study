# Predicting house prices using Linear Regression
from sklearn.linear_model import LinearRegression

X = [[50], [80], [100]]   # size in m²
y = [150, 240, 300]       # price in €k

model = LinearRegression().fit(X,y)
print(model.predict([[90]]))  # Predicting the price for a house of size 90 m²
# -----------------------------------------------------------------------------------------------------
# sizes and prices scaled down (size/100, price/100k) so training is stable
# Training data (scaled down so training is stable)
# size: m² / 100, price: € / 100k  (i.e. €k / 100)
X = [0.5, 0.8, 1.0]   # 50, 80, 100 m²
y = [1.5, 2.4, 3.0]   # 150k, 240k, 300k €

w, b = 0.0, 0.0       # start knowing nothing
lr = 0.1              # learning rate: size of each nudge

# Training: guess, measure the error, adjust, repeat
for step in range(5000):
    dw = db = 0
    for xi, yi in zip(X, y):
        error = (w * xi + b) - yi
        dw += 2 * error * xi / len(X)   # how the error changes with w
        db += 2 * error / len(X)        # how the error changes with b
    w -= lr * dw
    b -= lr * db

print(f"price = {w:.2f} * size + {b:.2f}")   # in scaled units


# Prediction: scale the input, predict, scale the result back
def predict(size_m2):
    size_scaled = size_m2 / 100
    price_scaled = w * size_scaled + b
    return price_scaled * 100                # price in €k

size = 70/100
print(f"Predicted price for {size} m²: {predict(size):.0f}k €")

# -----------------------------------------------------------------------------------------
def f(x, w):
    return w[0] * x + w[1]     # a line: slope * x + intercept

w = [3.0, 0.0]                 # learned parameters
print(f(90, w))                # ỹ = 270.0

# ------------------------------------------------------------------------------------------
# Predicting house prices using Linear Regression
from sklearn.linear_model import LinearRegression

X = [[50], [80], [100]]   # size in m²
y = [150, 240, 300]       # price in €k

model = LinearRegression().fit(X, y)
print(model.predict([[90]]))   # [270.]

# Learned parameters
w = model.coef_[0]        # slope
b = model.intercept_      # intercept

print("w =", w)
print("b =", b)
print(f"price = {w:.2f} * size + {b:.2f}")