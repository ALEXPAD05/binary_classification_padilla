# Program for binary classification 

import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import imdb # type: ignore
from keras import models, layers 
from tensorflow.keras.utils import plot_model #type: ignore

def binary_classification():
    """
    This function contains a neural network for binary classification.
    """
    # Load the IMDb dataset with the top 10,000 most frequent words
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print(train_data[0])  # Print the first training sample

    # Get the word index mapping words to integer indices
    word_index = imdb.get_word_index()

    list(word_index.items())[:5]  # Display first 5 word-index pairs

    # Reverse the word index to map indices back to words
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    list(reverse_word_index.items())[:5]  # Display first 5 index-word pairs

    # Decode the first training review back to text
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

    print(decoded_review)  # Print the decoded review

    print(train_labels[0])  # Print the first training label

    def vectorize_sequences(sequences, dimension=10000):
        """
        Converts sequences of integers into binary matrix representation.
        """
        results = np.zeros((len(sequences), dimension))  # Initialize zero matrix
        
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.  # Set 1 at corresponding word indices

        return results

    # Vectorize the training and test data
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    print(x_train[0])  # Print the first vectorized training sample

    # Convert labels to float32 numpy arrays
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # Define the neural network model
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))  # First hidden layer
    model.add(layers.Dense(16, activation='relu'))  # Second hidden layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Generate model architecture visualization
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Create validation set
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    # Compile the model
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    history_dict = history.history  # Store training history
    history_dict.keys()  # Display available training metrics

    loss_values = history_dict['loss']  # Training loss values
    val_loss_values = history_dict['val_loss']  # Validation loss values

    epochs = range(1, len(loss_values) + 1)  # Define epoch range

    # Plot training and validation loss
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()  # Clear the figure

    acc_values = history_dict['accuracy']  # Training accuracy values
    val_acc_values = history_dict['val_accuracy']  # Validation accuracy values

    # Plot training and validation accuracy
    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')

    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Evaluate the model on test data
    model.evaluate(x_test, y_test)
    results = model.evaluate(x_test, y_test)
    print(results)  # Print evaluation results
    model.predict(x_test[0:2])  # Make predictions on first two test samples

    plt.show()  # Show plots
