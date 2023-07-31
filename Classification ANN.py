import pandas
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('./data/train.csv')
color_dict = {'red': 0, 'blue': 1, 'green': 2, 'teal': 3, 'orange': 4, 'purple': 5}
train_data['color'] = train_data.color.apply(lambda x: color_dict[x])
np.random.shuffle(train_data.values)

x = train_data['x'].values
y = train_data['y'].values

plt.scatter(x[0:1000], y[0:1000], c='red', s=5)
plt.scatter(x[1001:2000], y[1001:2000], c='blue', s=5)
plt.scatter(x[2001:3000], y[2001:3000], c='green', s=5)
plt.scatter(x[3001:4000], y[3001:4000], c='teal', s=5)
plt.scatter(x[4001:5000], y[4001:5000], c='orange', s=5)
plt.scatter(x[5001:6000], y[5001:6000], c='purple', s=5)
plt.show()


model = keras.Sequential([
	keras.layers.Dense(32, input_shape=(2,), activation='relu'), keras.layers.Dense(32, activation='relu'), keras.layers.Dense(6, activation='sigmoid')])

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

x = np.column_stack((train_data.x.values, train_data.y.values))

model.fit(x, train_data.color.values, batch_size=1, epochs=10)

test_data = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_data.x.values, test_data.y.values))

print("EVALUATION")
test_data['color'] = test_data.color.apply(lambda x: color_dict[x])
model.evaluate(test_x, test_data.color.values)
