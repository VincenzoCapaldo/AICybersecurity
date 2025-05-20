import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import *
from test_set import get_test_set

def get_adversarial_images(images_dir, num_samples):
    # Raccolta di tutti i file .npy nelle varie sottocartelle
    all_npy_files = []
    files = [f for f in os.listdir(images_dir) if f.endswith(".npy")]
    npy_files = [os.path.join(images_dir, f) for f in sorted(files)]
    all_npy_files.extend(npy_files)

    # Calcolo di quanti campioni prelevare da ogni file
    samples_per_file = num_samples // len(all_npy_files)
    remainder = num_samples % len(all_npy_files)

    # Prelievo di campioni casuali da file .npy
    imgs_subset = []
    for i, npy_file in enumerate(all_npy_files):
        data = np.load(npy_file)
        n_samples = samples_per_file + (1 if i < remainder else 0) # distribuisce i campioni rimanenti
        indices = np.random.choice(data.shape[0], size=n_samples, replace=False)
        imgs_subset.append(data[indices])
    
    # Concatenzione dei subset di immagini
    imgs = np.concatenate(imgs_subset, axis=0).reshape(-1, 3, 224, 224)
    return imgs

def main():
    np.random.seed(33)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NUM_SAMPLES_ADVERSARIAL = 1000 # numero di campioni adversarial da inserire nel test set (1000 come i campioni clean)
    
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Caricamento dei detectors
    detectors = load_detectors(attack_types, device)

    # Caricamento delle immagini clean (e delle rispettive etichette) del test set
    clean_images, _ = get_test_set().get_images()
    clean_labels = np.zeros(len(clean_images), dtype=bool)
    
    for attack_type in attack_types:
        # Caricamento delle immagini adversarial del test set:
        images_dir = "./dataset/test_set/adversarial_examples/" + attack_type
        if attack_type == "df": # nel test set per il detector di DeepFool vengono usati 1000 campioni adversarial untargeted:
            images_dir1 = images_dir + "/untargeted/samples_plot1"
            imgs_adv = get_adversarial_images(images_dir1, NUM_SAMPLES_ADVERSARIAL)
        else: # nel test set di tutti gli altri attacchi vengono usati 1000 campioni adversarial (500 untargeted e 500 untarget):
            images_dir1 = images_dir + "/untargeted/samples_plot1"
            images_dir2 = images_dir + "/targeted/samples_plot1"
            imgs_adv1 = get_adversarial_images(images_dir1, NUM_SAMPLES_ADVERSARIAL//2)
            imgs_adv2 = get_adversarial_images(images_dir2, NUM_SAMPLES_ADVERSARIAL//2)
            imgs_adv = np.concatenate((imgs_adv1, imgs_adv2), axis=0)
        # Creazione delle etichette associate alle immagini adversarial:
        adv_labels = np.ones(len(imgs_adv), dtype=bool)

        # Creazione del test set complessivo (composto da immagini clean e da immagini adversarial)
        final_test_set = np.concatenate((clean_images, imgs_adv), axis=0)
        final_labels = np.concatenate((clean_labels, adv_labels), axis=0)
        
        # Chiamata al detector specifico dell'attacco
        detector = detectors[attack_type]
        report, is_adversarial = detector.detect(final_test_set)

        # Calcolo delle metriche di test
        y_true = final_labels
        y_pred = is_adversarial
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\nMetriche di test per il detector '{attack_type.upper()}':")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Crezione della directory in cui salvare i plot dei detectors
        plot_dir = "./plots/detectors/" + attack_type
        os.makedirs(plot_dir, exist_ok=True)

        # Crezione e salvataggio della matrice di confusione
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=["Clean", "Adversarial"])
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title(f"Confusion Matrix - {attack_type.upper()} Detector")
        plt.savefig(plot_dir + "/Confusion_Matrix.png")

        # Creazione e salvataggio della curva ROC
        logits = np.array(report["predictions"]) # contiene i logits delle classi "clean" e "adversarial"
        probs = softmax(logits, axis=1) # contiene le probabilità delle classi "clean" e "adversarial"
        probs_adv = probs[:, 1] # contiene solo le probabilità delle classi "adversarial"
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probs_adv)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure()
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f' ROC Curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve - {attack_type.upper()} Detector")
        plt.legend(loc='lower right')
        plt.savefig(plot_dir + "/ROC Curve.png")

if __name__ == "__main__":
    main()