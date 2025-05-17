import os
import pickle
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

NUM_CLASSES = 8631  # numero di classi nel dataset VGGFace2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # SENet
        compress_rate = 16
        # self.se_block = SEModule(planes * 4, compress_rate)  # this is not used.
        self.conv4 = nn.Conv2d(planes * 4, planes * 4 // compress_rate, kernel_size=1, stride=1, bias=True)
        self.conv5 = nn.Conv2d(planes * 4 // compress_rate, planes * 4, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        ## senet
        out2 = F.avg_pool2d(out, kernel_size=out.size(2))
        out2 = self.conv4(out2)
        out2 = self.relu(out2)
        out2 = self.conv5(out2)
        out2 = self.sigmoid(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        #out += residual
        out = out2 * out + residual
        out = self.relu(out)

        return out

# Definizione del modello SENet
class SENet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.include_top = include_top
        
        # Layer iniziali
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # Blocchi principali
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def senet50(**kwargs):
    """Constructs a SENet-50 model.
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def load_state_dict(model, model_path):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        model_path: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(model_path, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

# funzione per caricare il modello InceptionResnetV1 (NN1)
def get_NN1(device="cpu", classify=True):
    NN1 = InceptionResnetV1(pretrained='vggface2').eval()
    NN1.to(device)
    NN1.classify = classify
    print("Modello NN1 caricato correttamente")
    return NN1

# funzione per caricare il modello SENet (NN2)
def get_NN2(device="cpu", model_path='./models/senet50_ft_weight.pkl'):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model = senet50(num_classes=8631, include_top=True)
    load_state_dict(model,model_path)
    model.to(device)
    model.eval()
    print("Modello NN2 caricato correttamente da", model_path)
    return model

# Rete per rilevare immagini avversarie (adv)
class AdversarialDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # output: [clean, adversarial]
        )

    def forward(self, x):
        feats = self.backbone(x)  # [B, 512, 1, 1]
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)
    
    def fit(self, x_train, y_train, nb_epochs=40, batch_size=16, verbose=True,
        lr=1e-4, device='cpu', patience=5):
    
        self.to(device)

        # Split 80/20
        dataset = TensorDataset(x_train, y_train)
        total_size = len(dataset)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in tqdm(range(nb_epochs), desc="Epochs"):
            self.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / train_size

            # === Validation ===
            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_loss /= val_size
            val_acc = correct / total

            if verbose:
                print(f"Epoch {epoch+1}/{nb_epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # === Early stopping ===
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Ripristina il modello migliore
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

# Restituisce una rete detector, con backbone InceptionResNet
def get_detector(device="cpu", finetune=False):

    if finetune:
        # Finetuning della rete
        backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
    else:
        # Addestra la rete da zero
        backbone = InceptionResnetV1(classify=False)
    
    # Sblocca tutta la backbone
    for param in backbone.parameters():
        param.requires_grad = True
    
    detector = AdversarialDetector(backbone)
    detector.to(device)
    return detector

# Setup per classificatore NN1
def setup_classifierNN1(device, classify=True):
    # Istanzio la rete
    nn1 = get_NN1(device, classify)
    # Definizione del classificatore
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
    return classifierNN1

# Setup per classificatore NN2
def setup_classifierNN2(device):
    # Istanzio la rete
    nn2 = get_NN2(device)
    classifierNN2 = PyTorchClassifier(
        model=nn2,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(nn2.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        clip_values=(0.0, 255.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return classifierNN2

# Setup per classificatore del detector
def setup_detector_classifier(device):
    # Istanzio la rete
    detector = get_detector(device)
    # Definizione del classificatore
    classifier = PyTorchClassifier(
        model=detector,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(detector.parameters(), lr=1e-4),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=2,
        clip_values=(-1.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return classifier
