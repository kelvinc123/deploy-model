# Model to predict the image of handwritten digit (0-9) to the digit
import tensorflow as tf


# Code to load dataset from tensorflow library
# The dataset is handwritten digit (0 to 9)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# This code is just normalizing the input, basically making the data cleaner
x_train, x_test = x_train / 255.0, x_test / 255.0


# This part is defining the model
# The model has 2 layers: 128 neurons and 10 neurons
# The last layer is 10 neurons because we want to predict one number between 0 to 9
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),  # This is the middle layer that has 128 neurons, you can modify it as you like (10, 1000, 2000, etc) this is called hyperparameter
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)  # This is the last layer
])


# Loss function: Every model has a loss function and the ML objective is to minimize the loss function
# you can think of the loss as how bad the model is.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# Every ML model also has an optimizer, we need the optimizer to minimize the loss function
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# THis code is to train the model, we train it for 5 iteration, meaning that we update the model using the entier dataset 5 times
# you can change 5 to any number you like, try to make it more accurate
model.fit(x_train, y_train, epochs=5)

# This code is to evaluate the accuracy
model.evaluate(x_test,  y_test, verbose=2)

# Save model as h5 file
model.save('model1.h5')