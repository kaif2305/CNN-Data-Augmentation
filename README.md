# Enhanced CIFAR-10 Classification with CNN and Data Augmentation (TensorFlow/Keras)

This project presents an advanced Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset, incorporating best practices such as data augmentation and Batch Normalization to improve model performance and generalization.

## Project Overview

The Python script demonstrates a robust approach to image classification using TensorFlow/Keras:

1.  **Dataset Loading & Preprocessing**: Loads the CIFAR-10 dataset and normalizes pixel values.
2.  **Data Augmentation**: Sets up an `ImageDataGenerator` for on-the-fly image transformations to expand the training dataset and improve model robustness.
3.  **Advanced CNN Architecture**: Defines a deeper CNN model incorporating multiple convolutional blocks, Batch Normalization, MaxPooling, and Dropout layers.
4.  **Model Compilation**: Configures the model with an Adam optimizer and categorical cross-entropy loss.
5.  **Model Training**: Trains the CNN using the augmented data generator.
6.  **Model Evaluation**: Assesses the trained model's performance on the unseen test dataset.
7.  **Visualization**: Plots the training/validation accuracy and loss curves to analyze model learning and convergence.

## Dataset

The **CIFAR-10 dataset** is a standard benchmark in computer vision, consisting of 60,000 32x32 color images across 10 distinct classes (e.g., airplanes, automobiles, birds, cats). It's split into 50,000 training images and 10,000 test images.

### Data Preprocessing

* **Normalization**: Pixel values are scaled from the range `[0, 255]` to `[0.0, 1.0]`. This is a common practice to ensure numerical stability during training.
* **One-Hot Encoding**: Labels are converted into a one-hot encoded format (e.g., class 3 becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`), which is required for `categorical_crossentropy` loss.

### Data Augmentation

An `ImageDataGenerator` is used to create new, modified versions of the training images on the fly during training. This artificially increases the diversity of the training data without needing to store new images, helping the model generalize better and reduce overfitting. The augmentations applied include:

* `rotation_range=15`: Randomly rotates images by up to 15 degrees.
* `width_shift_range=0.1`: Randomly shifts images horizontally by up to 10% of the total width.
* `height_shift_range=0.1`: Randomly shifts images vertically by up to 10% of the total height.
* `horizontal_flip=True`: Randomly flips images horizontally.

## Advanced CNN Architecture

The `create_model()` function defines a Sequential CNN model designed for better performance on image classification tasks:

```python
def create_model():
    model = models.Sequential()
    
    # Convolutional Block 1
    model.add(layers.Input(shape=(32, 32, 3))) # Input layer for 32x32 RGB images
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization()) # Normalizes activations from the previous layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2))) # Downsampling
    model.add(layers.Dropout(0.25)) # Randomly sets 25% of input units to 0 to prevent overfitting
    
    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Fully connected layers (Classifier Head)
    model.add(layers.Flatten()) # Flattens the 2D feature maps to a 1D vector
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5)) # More aggressive dropout for the dense layer
    model.add(layers.Dense(10, activation='softmax')) # Output layer for 10 classes
    
    return model