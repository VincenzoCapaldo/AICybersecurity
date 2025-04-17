import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import transforms


def compute_accuracy(classifier, x_test, y_test):
    # Predizioni del modello (output con le probabilità per ogni classe)
    y_pred = classifier.predict(x_test)  # Shape: (N, 8631)

    # Convertiamo da probabilità a etichette (argmax sulle colonne)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predizioni finali

    # Calcoliamo l'accuratezza
    accuracy = accuracy_score(y_pred_labels, y_test)
    return accuracy

    
def process_image(image, target_size=(224, 224), use_padding=True):
    if use_padding:
        current_height, current_width = image.shape[1], image.shape[2]
        pad_height = (target_size[0] - current_height) // 2
        pad_width = (target_size[1] - current_width) // 2
        padded_image = F.pad(image, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)
        return padded_image
    else:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        return transform(image)