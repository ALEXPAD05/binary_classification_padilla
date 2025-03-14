# Binary Classification Model

This project implements a binary classification model using a neural network. It consists of two main scripts:

1. **`B_C_NLP.py`**: Defines and trains the binary classification model using the IMDb dataset.
2. **`main.py`**: Serves as the entry point for executing the classification model.

## Requirements

Ensure you have the following dependencies installed before running the scripts:

```bash
pip install numpy matplotlib keras tensorflow
```

## File Descriptions

### `B_C_NLP.py`

This script contains the `binary_classification()` function, which:
- Loads the IMDb dataset, limited to the 10,000 most frequent words.
- Converts movie reviews into binary vector representations.
- Defines a neural network model with:
  - An input layer with 10,000 features.
  - Two hidden layers with ReLU activation.
  - An output layer with a sigmoid activation for binary classification.
- Splits the data into training and validation sets.
- Compiles and trains the model using the RMSprop optimizer and binary cross-entropy loss.
- Plots training loss and accuracy.
- Evaluates the model on test data.

### `main.py`

This script serves as the main entry point for running the model. It:
- Imports `binary_classification` from `B_C_NLP.py`.
- Prints a message indicating the start of training.
- Calls `binary_classification()` to train and evaluate the model.
- Prints a message upon completion.

## How to Run

1. Ensure all dependencies are installed.
2. Run the `main.py` script:

```bash
python main.py
```

This will start training the binary classification model and display relevant outputs, including loss and accuracy plots.

## Output

- The trained model performance is evaluated on test data.
- The loss and accuracy trends are plotted for training and validation phases.
- The `model_plot.png` file is generated to visualize the network architecture.

## License

This project is open-source and can be modified or distributed freely.

