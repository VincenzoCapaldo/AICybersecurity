import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt


def compute_accuracy(classifier, x_test, y_test):
    # Predizioni del modello (output con le probabilità per ogni classe)
    y_pred = classifier.predict(x_test)  # Shape: (N, 8631)

    # Convertiamo da probabilità a etichette (argmax sulle colonne)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predizioni finali

    # Calcoliamo l'accuratezza
    accuracy = accuracy_score(y_pred_labels, y_test)
    return accuracy


def process_images(images, target_size=(224, 224), use_padding=True):
    processed_images = []
    resize = transforms.Resize(target_size)
    
    for image in images:
        image = torch.from_numpy(image)  # Converti in tensore PyTorch
        if use_padding:
            current_height, current_width = image.shape[1], image.shape[2]
            pad_height = (target_size[0] - current_height) // 2
            pad_width = (target_size[1] - current_width) // 2
            padded_image = F.pad(image, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)
            processed_images.append(padded_image.numpy())  # Converti in NumPy
        else:
            resized_image = resize(image)  # Applica Resize
            processed_images.append(np.array(resized_image))  # Converti in NumPy

    return np.stack(processed_images, axis=0)


def show_image(image):
    # Trasponiamo da (C, W, H) a (W, H, C) per matplotlib
    image = np.transpose(image, (1, 2, 0))

    # Mostriamo l'immagine
    plt.imshow(image)
    plt.axis('off')  # per togliere gli assi
    plt.show()

def plot_accuracy(title, x_title, x, average_perturbations, accuracies, targeted=False, targeted_accuracies=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle(title, fontsize=16)

    # Accuracy and Targeted Accuracy vs x
    axes[0].plot(x, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[0].plot(x, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[0].legend(["Accuracy", "Targeted Accuracy"])
    else:
        axes[0].legend(["Accuracy"])
    axes[0].set_xlabel(x_title)
    axes[0].grid()

    # Accuracy and Targeted Accuracy vs Average Perturbation
    axes[1].plot(average_perturbations, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[1].plot(average_perturbations, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[1].legend(["Accuracy", "Targeted Accuracy"])
    else:
        axes[1].legend(["Accuracy"])
    axes[1].set_xlabel("Average Perturbation")
    axes[1].grid()

    plt.tight_layout()
    filename = f"{title}.png"
    save_path = os.path.join("./plot", filename)
    plt.savefig(save_path)
    print(f"Plot {title}.png salvato.")