# %% 
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import Sequential, layers, applications
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, datetime

# %%
# 1. Data Loading
PATH = os.path.join(os.getcwd(), 'dataset')

# %%
# 2. Data Preparation
# Define batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (160,160)
SEED = 12345

# Load the data as tensorflow dataset using the special method
train_dataset = keras.utils.image_dataset_from_directory(PATH, batch_size=BATCH_SIZE, image_size=IMG_SIZE, seed=SEED, validation_split=0.3, subset='training')

val_dataset = keras.utils.image_dataset_from_directory(PATH, batch_size=BATCH_SIZE, image_size=IMG_SIZE, seed=SEED, validation_split=0.3, subset='validation')

# %%
# 3. Display some examples
# Extract class name as a list
class_names = train_dataset.class_names

# Plot some examples
plt.figure(figsize=(5,5))
for images, labels in train_dataset.take(1) :
    for i in range(9) :
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# %%
# 4. Perform validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
# 5. Convert the train, validation, test dasaset into prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
# 6. Create a 'model' for image augmentation
data_augmentation = Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
# 7. Repeatedly apply data augmentation on one image and see the result
for images, labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(5,5))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

# %%
# 8. Create the layer for data normalization
preprocess_input = applications.mobilenet_v2.preprocess_input

# %%
# 9. Transfer learning
# i. Instantiate the pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# %%
# ii. Set the pretrained model as non-trainable(frozen)
base_model.trainable = False
base_model.summary()
keras.utils.plot_model(base_model, show_shapes=True)

# %%
# iii. Create the classifier
# Create the global average pooling layer
global_avg = layers.GlobalAveragePooling2D()

# Create an output layer
output_layer = layers.Dense(len(class_names), activation='softmax')

# %%
# 10. Link the layers together to form a pipeline
inputs = keras.Input(shape=IMG_SHAPE)

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg(x)
x = layers.Dropout(0.3)(x)

outputs = output_layer(x)

# Instantitate the full model pipeline
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

# %%
# 11. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# 12. Evaluate the model
LOG_DIR = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=LOG_DIR)
es = EarlyStopping(patience=5, monitor='val_accuracy')

history = model.fit(pf_train, validation_data=pf_val, epochs=10, callbacks=[tb,es])

# %%
# 13. Model deployment
# Use the model to perform prediction
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch), axis=1)

# Classification report
print(classification_report(label_batch, y_pred))

# %%
# save model
model.save('model.h5')