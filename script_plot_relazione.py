import argparse
import torch
from nets import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod
from PIL import Image
from torchvision import transforms

NUM_CLASSES = 8631  # numero di classi nel dataset VGGFace2
LABELS = np.load("./dataset/rcmalli_vggface_labels_v2.npy")
# Trasformazione per adattare al classificatore
transform = transforms.Compose([
    transforms.ToTensor(),  # output: [C, H, W] con valori in [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # per portare in [-1, 1]
])


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on a classifier.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup modello
    nn1 = get_NN1("cuda")
    classifierNN1 = PyTorchClassifier(
        model=nn1,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(nn1.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        clip_values=(-1.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )

    # Trasformazione immagine per attacco
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
    ])

    # Carica immagine
    image_path = "./dataset/test_set/clean/processed/n006763/n006763_04.jpg"
    image = Image.open(image_path).convert("RGB")

    # Prepara batch
    image_tensor = transform(image)  # [3, 224, 224]

    
    image_np = np.expand_dims(image_tensor.numpy(), axis=0)  # [1, 3, 224, 224]
    
    clean_pred = nn1(image_tensor.unsqueeze(0).to(device))
    clean_pred = LABELS[np.array(clean_pred[0].detach().cpu().numpy()).argmax()]
    print(clean_pred)

    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    print(f"Selected attacks: {attack_types}")
    
    titles = ["./plot_attacchi_relazione/fgsm_plot_esempio.png",
              "./plot_attacchi_relazione/bim_plot_esempio.png",
              "./plot_attacchi_relazione/pgd_plot_esempio.png",
              "./plot_attacchi_relazione/df_plot_esempio.png",
              "./plot_attacchi_relazione/cw_plot_esempio.png"
              ]
    values_to_iterate = [[0.02, 0.04, 0.08, 0.1],
                        [0.02, 0.04, 0.08, 0.1],
                        [0.02, 0.04, 0.08, 0.1],
                        [1e-4, 1e-2, 1e-1, 1],
                        [0.001, 0.01, 0.1, 1]
                        ]

    for i, attack_type in enumerate(attack_types):
        attacked_images = []
        adv_class = []
        for values in values_to_iterate[i]:
            if attack_type == "fgsm":
                attack = FastGradientMethod(estimator=classifierNN1, eps=values, targeted=False)
            elif attack_type == "bim":
                attack = BasicIterativeMethod(estimator=classifierNN1, eps=values, eps_step=0.01, max_iter=20)
            elif attack_type == "pgd":
                attack = ProjectedGradientDescent(estimator=classifierNN1, eps=values, eps_step=0.01, max_iter=20)
            elif attack_type == "df":
                attack = DeepFool(classifier=classifierNN1, epsilon=values, nb_grads=40, max_iter=20)
            elif attack_type == "cw":
                attack = CarliniLInfMethod(classifier=classifierNN1, confidence=values, max_iter=20)
            adv = attack.generate(image_np)  # [1, 3, 224, 224]
            attacked_images.append(adv)
            pred_adv = nn1(torch.tensor(adv).to(device))
            pred_adv = pred_adv.argmax(dim=1).item()
            adv_class.append(LABELS[pred_adv])
        # Plot
        n = len(values_to_iterate[i]) + 1
        plt.figure(figsize=(2.5 * n, 6))

        # Immagine originale (prima riga)
        plt.subplot(2, n, 1)
        orig_disp = ((image_tensor.numpy().transpose(1, 2, 0) + 1.0) / 2.0)
        plt.imshow(np.clip(orig_disp, 0, 1))
        plt.title(f"Originale \n Pred: {clean_pred}")
        plt.axis('off')

        # Immagini adversarial (prima riga)
        for j, (img_adv, eps) in enumerate(zip(attacked_images, values_to_iterate[i]), start=2):
            adv_disp = ((img_adv.squeeze().transpose(1, 2, 0) + 1.0) / 2.0)
            plt.subplot(2, n, j)
            plt.imshow(np.clip(adv_disp, 0, 1))
            if attack_type == "cw":
                plt.title(f"confidence = {eps}\nPred: {adv_class[j-2]}")
            else:
                plt.title(f"eps = {eps}\nPred: {adv_class[j-2]}")
            plt.axis('off')

        # Rumori (seconda riga, a partire dalla seconda colonna)
        image_np_base = image_np[0]  # shape: [3, 224, 224]
        for j, img_adv in enumerate(attacked_images, start=2):  # start=2 per saltare la prima colonna
            noise = img_adv[0] - image_np_base
            noise_disp = (noise.transpose(1, 2, 0) + 1.0) / 2.0
            plt.subplot(2, n, n + j)
            plt.imshow(np.clip(noise_disp, 0, 1))
            plt.title("Noise")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(titles[i])

if __name__ == "__main__":
    main()