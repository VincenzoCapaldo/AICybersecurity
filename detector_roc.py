import argparse
import numpy as np
import os
from nets import setup_classifierNN1, setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector
import torch
from dataset import get_test_set, get_train_set
from utils import *
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw


def main():

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory per i modelli
    os.makedirs("./models", exist_ok=True)

    # Train or load Detectors
    detectors = {}
    # indica i detector da addestrare o caricare
    #attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    attack_types = ["fgsm"]
    

    # Carica le immagini e le etichette del test set
    test_set = get_test_set()
    test_images, _ = test_set.get_images()
    #prendo 20 campioni casuali clean
    num_samples = 1000
    total_samples = len(test_images)
    indices = np.random.choice(total_samples, num_samples, replace=False)
    # Seleziona le immagini e le etichette corrispondenti
    sampled_images = np.array(test_images[indices])
    clean_labels = np.zeros(len(sampled_images), dtype=bool)

    for attack_type in attack_types:
        model_path = os.path.join("./models", f"{attack_type}_detector.pth")
        detector_classifier = setup_detector_classifier(device)
        detector_classifier.model.load_state_dict(torch.load(model_path, map_location=device))
        detector_classifier.model.eval()
        detectors[attack_type] = BinaryInputDetector(detector_classifier)
        print(f"Detector caricato da: {model_path}")

        # solo per prova
        images_dir = "./dataset/test_set/adversarial_examples/" + attack_type + "/untargeted"
        #target_dir = "targeted" if targeted else "untargeted"
        imgs_adv = load_images_from_npy_folder(images_dir)
        imgs_adv = np.array(imgs_adv).reshape(-1, 3, 224, 224)
        selected = []

        # seleziono 200 immagini per ogni attacco
        for i in range(0, imgs_adv.shape[0], 1000):
            selected.append(imgs_adv[i:i+200])
        imgs_adv = np.concatenate(selected, axis=0)
        
        print(f"Shape dati clean: {np.shape(sampled_images)} \nShape dati adversarial: {np.shape(imgs_adv)}")
        adv_labels = np.ones(len(imgs_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
        final_test = np.concatenate((imgs_adv, sampled_images), axis=0)
        #print(f"final test shape: {np.shape(final_test)}")
        final_labels = np.concatenate((adv_labels, clean_labels), axis=0)
        report, _ = detectors[attack_type].detect(final_test)
        logits = np.array(report["predictions"])  # shape (n_samples, 2)
        probs = softmax(logits, axis=1)
        probs = probs[:, 1]
        #print(f"Final test shape: {final_test.shape}, final labels shape: {final_labels.shape}, probs shape: {probs.shape}")
        best_threshold = compute_roc_curve(final_labels, probs, "FGSM_ROC", save_plot=True, show_plot=False)

        print(f"Best threshold for FGSM detector is: {best_threshold}")






if __name__ == "__main__":
    main()