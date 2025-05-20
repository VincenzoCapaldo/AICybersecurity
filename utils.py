import os
import numpy as np

# Funzione per processare le immagini dalla rete NN1 alla rete NN2
def process_images(images):
    processed_images = []
    mean_bgr = np.array([91.4953, 103.8827, 131.0912]).reshape(3, 1, 1)
    
    for image in images:
        image = (image + 1.0)  * (255.0 / 2)  # float32 [-1, 1] -> [0, 255]
        image = image[[2, 1, 0], :, :]  # RGB → BGR
        image -= mean_bgr  # normalizzazione
        processed_images.append(image)
        
    # restituisce immagini pronte per la seconda rete
    return np.stack(processed_images, axis=0)

# salva un array di immagini in un file '.npy' in una cartella specifica
def save_images_as_npy(images, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = filename.replace(".", ",")
    filepath = os.path.join(save_dir, f"{filename}.npy")
    np.save(filepath, images)  # salva l'intero array di immagini in un unico file

def load_images_from_npy(folder_path):
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

# carica e restituisce un dizionario di detector ART salvati in locale 
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