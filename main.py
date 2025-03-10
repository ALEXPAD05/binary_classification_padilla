# Import the function `binary_classification` from the `B_C_NLPl` module in the `src` package
from src.B_C_NLP import binary_classification

def main():
    """
    Main entry point of the program.
    """
    # Print a message indicating the start of the binary classification model training
    print("Iniciando el entrenamiento de clasificacion binaria...")
    
    # Call the function to train the MNIST model
    binary_classification()
    
    # Print a message indicating the completion of the training
    print("Entrenamiento completado.")

# Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    # Call the main function to start the program
    main()
