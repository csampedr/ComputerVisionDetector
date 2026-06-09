import tensorflow as tf
from tensorflow import keras
import numpy as npprint("Cargando datos MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0                                               /home/csampedr/CPE4903/train_mnist.py
x_test  = x_test.astype('float32') / 255.0

# Reshape para CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

print("Construyendo modelo...")
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("Entrenando... (puede tardar 20-30 minutos en la Pi)")
model.fit(x_train, y_train,
          epochs=5,
          batch_size=64,
          validation_data=(x_test, y_test))

print("Guardando modelo...")
model.save('model_mnist_cnn.h5')
print("Modelo guardado como model_mnist_cnn.h5")

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Precisión final: {acc*100:.2f}%")
