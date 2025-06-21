import tensorflow as tf
import numpy as np

x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y = tf.constant([[2.0], [4.0], [6.0], [8.0]])

model = tf.keras.Sequential([tf.keras.layers.Dense(1)])

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, verbose=0)

prediction = model.predict([[5.0]])
print(prediction)
print("GPU used:", tf.config.list_physical_devices('GPU'))
