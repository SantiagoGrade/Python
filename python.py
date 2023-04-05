import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-42, -40, -37, -35, -33, -30, -28, -25, -22, -20, -15, -12, -8, -5, -3, -1, 0, 1, 3, 5, 8, 12, 15, 20, 22, 25, 28, 30, 33, 35, 37, 40, 42], dtype=float)
fahrenheit = np.array([-43.6, -40, -34.6, -31, -27.4, -22, -18.4, -13, -7.6, -4, 5, 10.4, 17.6, 23, 26.6, 30.2, 32, 33.8, 37.4, 41, 46.4, 53.6, 59, 68, 71.6, 77, 82.4, 86, 91.4, 95, 98.6, 104, 107.6], dtype=float)

capa = tf.keras.layers.Dense(units=4, input_shape=[1])
capa2 = tf.keras.layers.Dense(units=4, input_shape=[1])
oculta1 = tf.keras.layers.Dense(units=4, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=4, input_shape=[1])
oculta3 = tf.keras.layers.Dense(units=4, input_shape=[1])

salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa, capa2, oculta1, oculta2, oculta3, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="mean_squared_error"
)

print("comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000,verbose=False)
print("entrenada")

import matplotlib.pyplot as plt
plt.xlabel("$ Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print("Hagamos una prediccion")
resultado = modelo.predict([96])
print("El resultado es "+ str(resultado)+ " fahrenheit")
