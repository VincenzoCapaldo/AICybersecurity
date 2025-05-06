import argparse
import numpy as np
import os
from nets import setup_classifierNN1, setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector
import torch
from dataset import get_test_set, get_train_set
from utils import *
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw
import matplotlib.pyplot as plt

# data la cartella dell'attacco (test set), restituisce tutti i numpy al suo interno (l'unione di tutti gli attacchi (plot1, ...)))
def get_adversarial_images(images_dir, num_samples=1000):
    # carica tutte le sottocartelle
    subdirs = [os.path.join(images_dir, d) for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    # usato da fgsm che non ha sottocartelle
    imgs_adv = []
    if not subdirs:
        imgs_adv = load_images_from_npy_folder(images_dir)        
            
    # unisce tutte le immagini di quell'attacco (se target o non target viene passato dal parametro images_dir)
    for subdir in subdirs:
        imgs = load_images_from_npy_folder(subdir)
        imgs_adv.extend(imgs)

    imgs_adv = np.array(imgs_adv).reshape(-1, 3, 224, 224) # shape (num_campioni, 3, 224, 224)
    total_samples = imgs_adv.shape[0]
    indices = np.random.choice(total_samples, size=num_samples, replace=False)
    # Istogramma degli indici
    plt.hist(indices, bins=50, color='blue', edgecolor='black')
    plt.title("Distribuzione degli indici selezionati")
    plt.xlabel("Indice")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.show()
    imgs_subset = imgs_adv[indices]
    return imgs_subset



def main():

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory per i modelli
    os.makedirs("./models", exist_ok=True)

    #setta il seed per ripetere gli esperimenti
    np.random.seed(2025)
    NUM_SAMPLES_ADVERSARIAL = 1000  # numero di campioni da inserire nel test adversarial (dato che i dati clean sono 1000, usiamo 1000 campioni
                        # adversarial per avere un test bilanciato)        
    TEST = 3
    # Train or load Detectors
    detectors = {}
    # indica i detector da addestrare o caricare
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    attack_types = ["fgsm", "bim", "pgd"]
    attack_types = ["pgd"]
    

    # Carica le immagini e le etichette del test set
    test_set = get_test_set()
    clean_images, _ = test_set.get_images()
    clean_labels = np.zeros(len(clean_images), dtype=bool)
    # da usare se si vogliono plot sovrapposti (ancora non implementato)
    final_probs = []
    f_labels = []
    
    for attack_type in attack_types:
        print(f"Attacco: {attack_type}")
        model_path = os.path.join("./models", f"{attack_type}_detector.pth")
        detector_classifier = setup_detector_classifier(device)
        detector_classifier.model.load_state_dict(torch.load(model_path, map_location=device))
        detector_classifier.model.eval()
        detectors[attack_type] = BinaryInputDetector(detector_classifier)
        print(f"Detector caricato da: {model_path}")
        
        
        if TEST == 1:
            #### TEST 1 (50% clean - 50% untargeted) ####
            title = attack_type + "_ROC_untargeted"
            images_dir = "./dataset/test_set/adversarial_examples/" + attack_type + "/untargeted"
            imgs_adv = get_adversarial_images(images_dir, NUM_SAMPLES_ADVERSARIAL)
            
            
        if TEST == 2:
            #### TEST 2 (50% clean - 50% targeted) ####
            title = attack_type + "_ROC_targeted"
            images_dir = "./dataset/test_set/adversarial_examples/" + attack_type + "/targeted"
            imgs_adv = get_adversarial_images(images_dir, NUM_SAMPLES_ADVERSARIAL)

        if TEST == 3: 
            title = attack_type + "_ROC_targeted_untargeted"
            #### TEST 3 (50% clean - 25% untargeted - 25% targeted) ####
            images_dir1 = "./dataset/test_set/adversarial_examples/" + attack_type + "/untargeted"
            images_dir2 = "./dataset/test_set/adversarial_examples/" + attack_type + "/targeted"
            imgs_adv1 = get_adversarial_images(images_dir1, NUM_SAMPLES_ADVERSARIAL//2)
            imgs_adv2 = get_adversarial_images(images_dir2, NUM_SAMPLES_ADVERSARIAL//2)
            imgs_adv = np.concatenate((imgs_adv1, imgs_adv2), axis=0)

        print(f"{attack_type}: Shape dati clean: {np.shape(clean_images)}\tShape dati adversarial: {np.shape(imgs_adv)}")
        adv_labels = np.ones(len(imgs_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
        final_test = np.concatenate((imgs_adv, clean_images), axis=0)
        #print(f"final test shape: {np.shape(final_test)}")
        final_labels = np.concatenate((adv_labels, clean_labels), axis=0)
        report, _ = detectors[attack_type].detect(final_test)
        logits = np.array(report["predictions"])  # shape (n_samples, 2)
        probs = softmax(logits, axis=1)
        probs = probs[:, 1]
        #print(f"Final test shape: {final_test.shape}, final labels shape: {final_labels.shape}, probs shape: {probs.shape}")
        final_probs.append(probs)
        f_labels.append(final_labels)
        #print(f"shape final_probs: {np.shape(final_probs)}\nshape final labels: {np.shape(f_labels)}")
        compute_roc_curve(final_labels, probs, title, save_plot=True, show_plot=False)
        print(f"ROC curve {title} salvata")


if __name__ == "__main__":
    main()