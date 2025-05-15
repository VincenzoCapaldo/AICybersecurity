from abc import ABC, abstractmethod
import numpy as np
import torch
from utils import *
from tqdm import tqdm
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod

NUM_CLASSES = 8631  # numero di classi nel dataset VGGFace2

class AdversarialAttack(ABC):
    def __init__(self, classifierNN1):
        self.classifierNN1 = classifierNN1 # gli attacchi vengono effettuati sul classificatore NN1

    @abstractmethod
    def generate_images(self, images, targeted=False, target_class=0):
        pass

# Classe per la gestione dell'attacco FGSM (Fast Gradient Sign Method)
class FGSM(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, targeted=False, targeted_labels=None):
        attack = FastGradientMethod(estimator=self.classifierNN1, eps=epsilon, targeted=targeted)
        if targeted:
            # attacco targeted
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # attacco untargeted
            return attack.generate(images)    

# Classe per la gestione dell'attacco BIM (Basic Iterative Method)
class BIM(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = BasicIterativeMethod(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)
        if targeted:
            # attacco targeted
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # attacco untargeted
            return attack.generate(images)    

# Classe per la gestione dell'attacco PGD (Projected Gradient Descent)
class PGD(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = ProjectedGradientDescent(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)
        if targeted:
            # attacco targeted
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # attacco untargeted
            return attack.generate(images)

# Classe per la gestione dell'attacco DF (DeepFool)
class DF(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, nb_grads, max_iter):
        # attacco untargeted (la libreria ART non supporta l'attacco DeepFool targeted)
        attack = DeepFool(classifier=self.classifierNN1, epsilon=epsilon, nb_grads=nb_grads, max_iter=max_iter, verbose=False)
        # Per motivi di efficienza viene processata un'immagine alla volta
        test_images_adv = []
        batch_dim = 1
        for j in tqdm(range (0, len(images), batch_dim), desc="DeepFool"):
            batch = images[j:j+batch_dim]
            test_images_adv.append(attack.generate(batch))
        test_images_adv = np.concatenate(test_images_adv, axis=0)
        return test_images_adv

# Classe per la gestione dell'attacco CW (Carlini-Wagner)
class CW(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, confidence, learning_rate, max_iter, targeted=False, targeted_labels=None):
        attack = CarliniLInfMethod(classifier=self.classifierNN1, confidence=confidence, learning_rate=learning_rate, max_iter=max_iter, initial_const=0.1, targeted=targeted)
        if targeted:
            # attacco targeted
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # attacco untargeted
            return attack.generate(images)