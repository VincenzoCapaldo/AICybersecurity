import os
import numpy as np
import torch
from torch.autograd import no_grad
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt


def compute_accuracy(classifier, x_test, y_test):
    # Predizioni del modello (output con le probabilità per ogni classe)
    y_pred = classifier.predict(x_test)  # Shape: (N, 8631)

    # Convertiamo da probabilità a etichette (argmax sulle colonne)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predizioni finali
    #print("predizioni: ", y_pred_labels)

    # Calcoliamo l'accuratezza
    accuracy = accuracy_score(y_pred_labels, y_test)
    return accuracy


def process_images(images, target_size=(256), use_padding=False):
    processed_images = []
    mean_bgr = np.array([91.4953, 103.8827, 131.0912]).reshape(3, 1 ,1) # media dei canali BGR
    for image in images:
        image = torch.from_numpy((image*255).astype(np.uint8))  # Converti in tensore PyTorch
        if use_padding:
            current_height, current_width = image.shape[1], image.shape[2]
            pad_height = (target_size - current_height) // 2
            pad_width = (target_size - current_width) // 2
            processed_img = F.pad(image, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)
        else:
            processed_img = transforms.Resize(target_size)(image)    # Applica Resize
        processed_img = transforms.CenterCrop(224)(processed_img)
        processed_img = processed_img.numpy()
        processed_img = processed_img[[2, 1, 0], :, :].astype(np.float32) # RGB -> BGR
        processed_img -= mean_bgr
        processed_images.append(torch.from_numpy(processed_img).float()) 
        
    return torch.stack(processed_images, dim=0)


def show_image(image):
    # Trasponiamo da (C, W, H) a (W, H, C) per matplotlib
    image = np.transpose(image, (1, 2, 0))

    # Mostriamo l'immagine
    plt.imshow(image)
    plt.axis('off')  # per togliere gli assi
    plt.show()

def plot_accuracy(title, x_title, x, max_perturbations, accuracies, accuracy_clean, targeted_accuracy_clean=0.0, targeted=False, targeted_accuracies=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # Valori sui dati clean
    if not x_title == "Target Class":
        x.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies.insert(0, accuracy_clean)
        if targeted:
            targeted_accuracies.insert(0, targeted_accuracy_clean)

    # Accuracy and Targeted Accuracy vs x
    axes[0].plot(x, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[0].plot(x, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[0].legend(["Accuracy", "Targeted Accuracy"], loc="upper right")
    else:
        axes[0].legend(["Accuracy"], loc="upper right")
    axes[0].set_xlabel(x_title)
    axes[0].grid()

    # Accuracy and Targeted Accuracy vs Max Perturbations
    axes[1].plot(max_perturbations, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[1].plot(max_perturbations, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[1].legend(["Accuracy", "Targeted Accuracy"], loc="upper right")
    else:
        axes[1].legend(["Accuracy"], loc="upper right")
    axes[1].set_xlabel("Max Perturbations")
    axes[1].axvline(x=0.05, color='red', linestyle='--', linewidth=1.5) # vincolo da rispettare
    axes[1].grid()
    
    plt.tight_layout()
    filename = f"{title}.png"
    save_path = os.path.join("./plot", filename)
    plt.savefig(save_path)
    print(f"Plot {title}.png salvato.")