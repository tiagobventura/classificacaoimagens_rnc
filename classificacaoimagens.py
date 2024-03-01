# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:59:30 2024

@author: tiago
"""
#importação de bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

print('Classificação de imagens')

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

'''
roses = list(data_dir.glob('roses/*'))
image = PIL.Image.open(str(roses[3]))
image.show()
'''
#Parâmetros
batch_size = 32
img_height = 180
img_width = 180
#Criando conjunto de dados.
#base de treinamento
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

#base de validação
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

class_names = train_ds.class_names
print(class_names)

#Visualize os dados

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#Configurar o conjunto de dados para desempenh
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

'''
Padronização de dados
- Os valores do canal RGB estão na faixa[0, 255].
- Para rede neural os valores de entrada devem ser pequenos, padronizando os valores
para ficarem no intervalo de [0, 1], utilizando o método tf.keras.layers.Rescaling
'''
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y:(normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
#Notice the pixel values are now in `x[0,1]`.
print(np.min(first_image), np.max(first_image))

'''
Criando o modelo.
'''
num_classes = len(class_names)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                   img_width,
                                   3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    ])

model = Sequential([
   # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

'''
Procedimento para resolver overfitting
- Aumento de dados.(Código abaixo)
'''
'''
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                   img_width,
                                   3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    ])
'''
'''
Compilar o modelo
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''
Resumo do modelo
- Visualize todas as camadas da rede usando o metodo Model.summary
'''
model.summary()

'''
Treinando o modelo
'''
epochs=15
'''
- Guarda variável o treinamento.
'''
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )

'''
Visualize os resultados do treinamento
- Gráficos de perda e precisão nos conjuntos de treinamento e validação.
'''

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()


'''
Saida de dados com a primeira solução de overfitting
'''
'''
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
'''
'''
Testando o modelo.
'''
#https://images.pexels.com/photos/69776/tulips-bed-colorful-color-69776.jpeg
'''
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
'''
sunflower_url = "https://www.oficinadeervas.com.br/images/produtos/20190705_102406_dente-de-leao.jpg"
sunflower_path = tf.keras.utils.get_file('20190705_102406_dente-de-leao', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
      "This image most likely belongd yo {} with a {:.2f} precent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
)




































































































