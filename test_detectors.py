import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from test_set import get_test_set

def get_adversarial_images(images_dir, num_samples=1000):
    # trova tutte le sottocartelle
    subdirs = [os.path.join(images_dir, d) for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    
    # caso in cui non ci sono sottocartelle (es. FGSM)
    folders_to_process = subdirs if subdirs else [images_dir]

    # raccoglie tutti i file .npy
    all_npy_files = []
    for folder in folders_to_process:
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        npy_files = [os.path.join(folder, f) for f in sorted(files)]  # restituisce i path completi
        all_npy_files.extend(npy_files)

    if not all_npy_files:
        raise ValueError("Nessun file .npy trovato nella directory fornita.")

    # calcola quanti campioni prendere per ogni file
    samples_per_file = num_samples // len(all_npy_files)
    remainder = num_samples % len(all_npy_files)

    imgs_subset = []
    for i, npy_file in enumerate(all_npy_files):
        data = np.load(npy_file)
        n_samples = samples_per_file + (1 if i < remainder else 0)  # distribuisce equamente l'eventuale resto
        indices = np.random.choice(data.shape[0], size=n_samples, replace=False)
        imgs_subset.append(data[indices])

    imgs_subset = np.concatenate(imgs_subset, axis=0).reshape(-1, 3, 224, 224)
    return imgs_subset

def compute_roc_curve(true_label, model_predictions, attack, save_plot=False, show_plot=False, save_dir = "./plots/detectors"):
    # Titolo e nome file
    title = f"ROC Curve - {attack.upper()} Detector"
    file_name = f"{attack.upper()}_ROC.png"
    
    # Calcolo dei valori ROC
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
        plt.savefig(save_dir + file_name)

def main():
    np.random.seed(33)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Crezione della directory in cui salvare i plot dei detectors:
    plot_dir = "./plots/detectors/"
    os.makedirs(plot_dir, exist_ok=True)

    NUM_SAMPLES_ADVERSARIAL = 1000  # numero di campioni da inserire nel test adversarial (dato che i dati clean sono 1000, usiamo 1000 campioni)
    
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Carica i Detectors
    detectors = load_detectors(attack_types, device)

    # Carica le immagini e le etichette del test set
    clean_images, _ = get_test_set().get_images()
    clean_labels = np.zeros(len(clean_images), dtype=bool)
    
    # Test su 1000 campioni clean, 500 untargeted, 500 targeted (per deepfool 1000 untargeted, 1000 clean)
    for attack_type in attack_types:
        images_dir = "./dataset/test_set/adversarial_examples/" + attack_type

        # Deepfool ha solo untargeted
        if attack_type == "df":
            images_dir1 = images_dir + "/untargeted/samples_plot1"
            imgs_adv = get_adversarial_images(images_dir1, NUM_SAMPLES_ADVERSARIAL)
        else:
            if attack_type == "fgsm":
                images_dir1 = images_dir + "/untargeted"
                images_dir2 = images_dir + "/targeted"
            else:
                images_dir1 = images_dir + "/untargeted/samples_plot1"
                images_dir2 = images_dir + "/targeted/samples_plot1"
            imgs_adv1 = get_adversarial_images(images_dir1, NUM_SAMPLES_ADVERSARIAL//2)
            imgs_adv2 = get_adversarial_images(images_dir2, NUM_SAMPLES_ADVERSARIAL//2)
            imgs_adv = np.concatenate((imgs_adv1, imgs_adv2), axis=0)

        print(f"{attack_type}: Shape dati clean: {np.shape(clean_images)}\tShape dati adversarial: {np.shape(imgs_adv)}")
        adv_labels = np.ones(len(imgs_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
        final_test = np.concatenate((imgs_adv, clean_images), axis=0)
        final_labels = np.concatenate((adv_labels, clean_labels), axis=0)
        # Chiamata al detector con .detect
        detector = detectors[attack_type]
        report, preds = detector.detect(final_test)  # final_test è già un np.ndarray di shape (N, 3, 224, 224)
        preds = preds.astype(bool)

        # Calcolo metriche
        y_true = final_labels
        y_pred = preds
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\nRisultati per il detector '{attack_type.upper()}':")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("Report dettagliato:")
        print(classification_report(y_true, y_pred, target_names=["Clean", "Adversarial"], zero_division=0))
        # Plot e salvataggio matrice di confusione
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred),
                                      display_labels=["Clean", "Adversarial"])
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title(f"Confusion Matrix - {attack_type.upper()} Detector")
        plt.tight_layout()
        plt.savefig(plot_dir + f"{attack_type.upper()}_Conf_Matrix.png")
        plt.close()

        # Salvataggio ROC curve
        logits = np.array(report["predictions"])  # shape (n_samples, 2)
        probs = softmax(logits, axis=1)
        probs = probs[:, 1]
        compute_roc_curve(final_labels, probs, attack_type, save_plot=True, show_plot=False, save_dir=plot_dir)
        print(f"ROC curve {attack_type} salvata")

if __name__ == "__main__":
    main()