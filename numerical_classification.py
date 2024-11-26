import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Create a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   #input layer
    tf.keras.layers.Dense(128, activation='relu'),   #first hidden layer
    tf.keras.layers.Dense(128, activation='relu'),   #second hidden layer
    tf.keras.layers.Dense(10, activation='softmax')  #output layer, softmax for probability distribution
])

# Compile the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test, verbose=2)

print('Validation loss:', val_loss)
print('Validation accuracy:', val_acc)

# Save the model
#model.save('model.h5')

