import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.special import softmax
from PIL import Image

def compute_max_perturbation(test_images, test_images_adv):
    return np.max(np.abs(test_images_adv - test_images))

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