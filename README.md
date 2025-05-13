# AICybersecurity - Progetto 2025
# 🛡️ Robustezza di un Sistema di Face Recognition agli Adversarial Attacks

## 📚 Descrizione del Progetto

Il progetto ha lo scopo di valutare la sicurezza di un sistema di riconoscimento facciale (face recognition) contro attacchi adversariali. In particolare, viene analizzata la robustezza della rete NN1 e la trasferibilità degli attacchi su un secondo classificatore NN2. Sono inoltre implementate e testate strategie di difesa contro tali attacchi.



---

## ⚙️ Setup dell'Ambiente

Clona questo repository e installa le dipendenze eseguendo i comandi seguenti:

```bash
# 1. Crea e attiva l'ambiente Conda
conda create -n ai_cyber python=3.10 -y
conda activate ai_cyber

# 2. Installa PyTorch con supporto CUDA
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Verifica che CUDA sia disponibile
python -c "import torch; print(torch.cuda.is_available())"

# 4. Installa le librerie necessarie
conda install conda-forge::adversarial-robustness-toolbox[all]
conda install -c conda-forge packaging
conda install conda-forge::matplotlib
pip install facenet-pytorch --no-deps
conda install anaconda::scikit-learn
