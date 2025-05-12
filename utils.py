import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.special import softmax
from PIL import Image
from nets import setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector
import torch


def compute_max_perturbation(test_images, test_images_adv):
    return np.max(np.abs(test_images_adv - test_images))


def compute_accuracy(classifier, x_test, y_test):
    # Predizioni del modello (output con le probabilità per ogni classe)
    y_pred = classifier.predict(x_test)  # Shape: (N, 8631)

    # Convertiamo da probabilità a etichette (argmax sulle colonne)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predizioni finali

    # Calcoliamo l'accuratezza
    accuracy = accuracy_score(y_pred_labels, y_test)
    return accuracy


def compute_accuracy_with_detectors(classifier, x_test, y_test, y_adv, detectors, threshold=0.5, targeted=False, verbose=True):
    """
    Calcola l'accuracy penalizzando i falsi positivi dei detector.
    - classifier: il classificatore (con metodo predict).
    - x_test, y_test: dati NON adversariali.
    - y_adv: etichette per i campioni avversari (True/False).
    - detectors: dict di detector ART.
    - threshold: soglia per considerare un campione come avversario.
    Ritorna: (accuracy_effettiva, n_samples_utili, n_falsi_positivi)
    """
    # Maschera di campioni rifiutati da almeno un detector
    rejected_mask = np.zeros(x_test.shape[0], dtype=bool)

    for name, detector in detectors.items():
        report, _ = detector.detect(x_test)
        logits = np.array(report["predictions"])  # shape (n_samples, 2)
        probs = softmax(logits, axis=1)
        adversarial_probs = probs[:, 1]
        is_adversarial = adversarial_probs > threshold # True se avversario; False se pulito
        rejected_mask = np.logical_or(is_adversarial, rejected_mask)  # Un campione è scartato se almeno un detector lo scarta
        detection_error = np.logical_xor(is_adversarial, y_adv)
        if verbose:
            print(f"Detector {name} ha scartato {np.sum(is_adversarial)} campioni (soglia={threshold}).")
            print(f"Detector {name} ha rilevato erroneamente {np.sum(detection_error)} campioni (soglia={threshold}).")
            print(f"Detector {name} ha rilevato correttamente {x_test.shape[0] - np.sum(detection_error)} campioni (soglia={threshold}).")

    accepted_mask = np.logical_not(rejected_mask)  # Inverti la maschera: True se accettato, False se scartato
    
    # Campioni accettati = quelli che passeranno al classificatore
    x_pass = x_test[accepted_mask]
    y_pass = y_test[accepted_mask]

    # Predizioni del classificatore
    if x_pass.shape[0] > 0:
        y_pred = classifier.predict(x_pass)
        y_pred_labels = np.argmax(y_pred, axis=1)
        n_correct = np.sum(y_pred_labels == y_pass)  # campioni correttamente classificati
    else:
        n_correct = 0
        if verbose:
            print("Nessun campione accettato dai detector.")
    
    is_adversarial = ~accepted_mask  # Campioni avversari: quelli scartati dai detector

    # Falsi positivi: campioni puliti scartati dai detector (0,1)
    n_fp = np.sum(np.logical_and(is_adversarial, ~y_adv))
    
    # Campioni correttamente scartati (1,1)
    n_correct_discarded = np.sum(np.logical_and(is_adversarial, y_adv)) # Veri positivi
    
    # Accuracy: corrette / totale originario (quindi penalizza falsi positivi)
    n_total = y_test.shape[0]
    if targeted:
        accuracy = n_correct/n_total # I campioni scartati non vengono considerati perchè l'attacco non è andato a buon fine
    else:    
        accuracy = (n_correct + n_correct_discarded) / n_total

    return accuracy, n_fp


def load_detectors(attack_types, device):
    detectors = {}
    for attack_type in attack_types:
            model_path = os.path.join("./models", f"{attack_type}_detector.pth")
            detector_classifier = setup_detector_classifier(device)
            detector_classifier.model.load_state_dict(torch.load(model_path, map_location=device))
            detector_classifier.model.eval()
            detectors[attack_type] = BinaryInputDetector(detector_classifier)
            print(f"Detector caricato da: {model_path}")
    return detectors


# Funzione per processare le immagini dalla rete NN1 alla rete NN2
def process_images(images):
    processed_images = []
    mean_bgr = np.array([91.4953, 103.8827, 131.0912]).reshape(3, 1, 1)
    
    for image in images:
        image = (image + 1.0)  * (255.0 / 2)  # float32 [-1, 1] -> [0, 255]
        image = image[[2, 1, 0], :, :]  # RGB → BGR
        image -= mean_bgr  # normalizzazione
        processed_images.append(image)
        
    return np.stack(processed_images, axis=0)


def show_image(img, title=""):
    img = (img + 1) / 2.0  # normalizza da [-1, 1] a [0, 1]
    img = np.transpose(img, (1, 2, 0))  # da (C, H, W) a (H, W, C)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

    # Mostra la finestra e aspetta che venga chiusa
    plt.show(block=True)


def save_images_as_jpg(images, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = filename.replace(".", ",")
    for i, img_array in enumerate(images):
        img_array = np.transpose(img_array, (1, 2, 0)) 
        # Se float, scala a 0-255 e converti in uint8
        if img_array.dtype == np.float32 or img_array.max() <= 1.0:
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(save_dir, filename + f'_{i}.jpg'), format='JPG')


def save_images_as_npy(images, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = filename.replace(".", ",")
    filepath = os.path.join(save_dir, f"{filename}.npy")
    np.save(filepath, images)  # salva l'intero array di immagini in un unico file


def load_images_from_npy_folder(folder_path):
    # Trova tutti i file .npy nella cartella
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError("Nessun file .npy trovato nella cartella.")

    images_list = []
    for file_name in sorted(files):  # sorted per coerenza
        file_path = os.path.join(folder_path, file_name)
        images_array = np.load(file_path)
        images_list.append(images_array)

    return images_list


def compute_roc_curve(true_label, model_predictions, title="Roc curve", title_image="roc", save_plot=False, show_plot=False):
    # Calcolo dei valori ROC
    save_dir = "detectors_plot/"
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, thresholds = roc_curve(true_label, model_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f' ROC Curve (area = {roc_auc:.2f})')
    # commentare la riga sottostante se si vuole eliminare la linea (diagonale) del detector completamente casuale
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid()
    # roc curve plot
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_dir + title_image + ".png")


def plot_accuracy(title, x_title, x, max_perturbations, accuracies, attack_dir, targeted=False, targeted_accuracies=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    #print(x, max_perturbations, accuracies)
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
    axes[1].axvline(x=0.1, color='red', linestyle='--', linewidth=1.5) # vincolo da rispettare
    axes[1].grid()
    
    plt.tight_layout()
    filename = title.replace(".",",")+ ".png"
    plot_dir = os.path.join("./plot", attack_dir)
    save_path = os.path.join(plot_dir, filename)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot {title}.png salvato.")
    plt.close()


def show_images_from_npy_folder(folder_path):
    images = load_images_from_npy_folder(folder_path)

    image_set = images[0]  # shape: (1000, 3, 224, 224)

    for i, img in enumerate(image_set):
        show_image(img, f'Immagine {i+1}/{len(image_set)}')
    

if __name__ == "__main__":
    show_images_from_npy_folder("./dataset/test_set/adversarial_examples/df/plot1")