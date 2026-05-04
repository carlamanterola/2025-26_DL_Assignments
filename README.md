# 2025-26_DL_Assignments
Utilizar esta linea para instalar todo:
pip install -r requirements.txt

# Para GPU:
# Primero desinstala la versión actual
pip uninstall torch torchvision torchaudio

# Instala la versión con soporte CUDA 12.1 (la más estándar hoy)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Extensión para gráfica intel
pip install intel-extension-for-pytorch

# Para usar gráfica intel:
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install intel-extension-for-pytorch==2.5.0