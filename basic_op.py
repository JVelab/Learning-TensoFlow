import tensorflow as tf

# Crear tensores
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Operaciones básicas
c = tf.matmul(a, b)
print(c)