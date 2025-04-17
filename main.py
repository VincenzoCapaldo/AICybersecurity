from nets import get_NN1, get_NN2
import torch
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from dataset import get_test_set
from utils import compute_accuracy


NUM_CLASSES = 8631



def main(device):
    # Istanzio le reti
    nn1 = get_NN1(device)
    nn2 = get_NN2(device)

    # Definizione dei classificatori
    classifierNN1 = PyTorchClassifier(
        model=nn1,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(nn1.parameters(), lr=0.001),
        input_shape=(3, 160, 160),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    classifierNN2 = PyTorchClassifier(
        model=nn2,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(nn2.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )

    # Carico il test_set
    dataloader, test_images, test_labels = get_test_set()

    # calcolo delle performance dei classificatori sui dati clean
    accuracy_nn1_clean = compute_accuracy(classifierNN1, test_images, test_labels)
    print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")
    #accuracy_nn2_clean = compute_accuracy(classifierNN2, test_images, test_labels)
    #print(f"Accuracy del classificatore NN2 su dati clean: {accuracy_nn2_clean}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main(device)