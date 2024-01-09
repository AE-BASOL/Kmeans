## evaluation.py

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_image_paths, load_images, calculate_histograms


def calculate_accuracy(actual_labels, predicted_labels):
    correct = sum(act == pred for act, pred in zip(actual_labels, predicted_labels))
    return correct / len(actual_labels)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def show_sample_images(images, labels, predicted_labels, class_names):
    fig, axes = plt.subplots(len(class_names), 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, class_name in enumerate(class_names):
        correct_images = [img for img, label, pred in zip(images, labels, predicted_labels) if
                          label == pred == class_name]
        incorrect_images = [img for img, label, pred in zip(images, labels, predicted_labels) if
                            label == class_name and label != pred]

        # Show 5 correct images
        for j in range(min(5, len(correct_images))):
            ax = axes[i, 0]
            ax.imshow(correct_images[j])
            ax.set_title(f"Correct: {class_name}")
            ax.axis('off')

        # Show 1 incorrect image
        if incorrect_images:
            ax = axes[i, 1]
            ax.imshow(incorrect_images[0])
            ax.set_title(f"Incorrect: {class_name}")
            ax.axis('off')

    plt.show()


def evaluate_model(model, train_histograms, train_labels, valid_csv_path, valid_dir, class_names):
    # Train set performance
    train_predicted_labels = model.predict(train_histograms)
    train_cm = confusion_matrix(train_labels, train_predicted_labels)
    train_accuracy = calculate_accuracy(train_labels, train_predicted_labels)
    print(f"Training Accuracy: {train_accuracy}")
    plot_confusion_matrix(train_cm, class_names)

    # Validation set performance
    valid_image_paths = load_image_paths(valid_csv_path, valid_dir)
    valid_images = load_images(valid_image_paths)
    valid_histograms = calculate_histograms(valid_images)

    # Flatten the list of histograms for prediction
    valid_histograms_flattened = [hist for sublist in valid_histograms.values() for hist in sublist]
    valid_labels_flattened = []
    for color in class_names:
        valid_labels_flattened.extend([color] * sum(valid_images[color] == 1))

    valid_predicted_labels = model.predict(valid_histograms_flattened)
    valid_cm = confusion_matrix(valid_labels_flattened, valid_predicted_labels)
    valid_accuracy = calculate_accuracy(valid_labels_flattened, valid_predicted_labels)
    print(f"Validation Accuracy: {valid_accuracy}")
    plot_confusion_matrix(valid_cm, class_names)

    # Display some sample images with their predicted labels
    show_sample_images(valid_images, valid_labels_flattened, valid_predicted_labels, class_names)
