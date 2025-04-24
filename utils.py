import os
import numpy as np
import torch
from torch.autograd import no_grad
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.special import softmax

def compute_accuracy(classifier, x_test, y_test):
    # Predizioni del modello (output con le probabilità per ogni classe)
    y_pred = classifier.predict(x_test)  # Shape: (N, 8631)

    # Convertiamo da probabilità a etichette (argmax sulle colonne)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predizioni finali
    #print("predizioni: ", y_pred_labels)

    # Calcoliamo l'accuratezza
    accuracy = accuracy_score(y_pred_labels, y_test)
    return accuracy


def compute_accuracy_with_detectors(classifier, x_test, y_test, y_adv, detectors, threshold=0.5, targeted=False, verbose=False):
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


def train_test_split(images, labels, test_size=0.2, shuffle=True, random_seed=2025):
    """
    Divide immagini e etichette in train e test set.

    Parametri:
        images (np.ndarray): Array delle immagini.
        labels (np.ndarray): Array delle etichette.
        test_size (float): Percentuale del dataset da usare come test (es. 0.2 per 20%).
        shuffle (bool): Se True, mescola i dati prima dello split.
        random_seed (int, opzionale): Seed per riproducibilità.

    Ritorna:
        train_images, train_labels, test_images, test_labels
    """
    assert images.shape[0] == labels.shape[0], "Numero di immagini ed etichette non corrisponde."
    
    num_samples = images.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_images = images[train_idx]
    train_labels = labels[train_idx]
    test_images = images[test_idx]
    test_labels = labels[test_idx]

    return train_images, train_labels, test_images, test_labels