import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt 

banco_de_imagens = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = banco_de_imagens.load_data()

class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metric=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

predictions = model.predict(test_images)

print(np.argmax(predictions[5]))

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(train_images[5])
plt.grid(False)
plt.show()