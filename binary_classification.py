# 1. Importar
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 2. Definir datos
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
X_test = np.random.rand(200, 10)
y_test = np.random.randint(0, 2, 200)

# Convertir a tensores
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

# 3. Construir modelo
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 4. Entrenar
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 5. Evaluar
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en prueba: {test_loss:.4f}, Precisión en prueba: {test_accuracy:.4f}")