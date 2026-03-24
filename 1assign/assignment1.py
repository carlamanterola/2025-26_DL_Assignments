#NECESSARY IMPORTS
import numpy as np
import pandas as pd
 
import torch
import torch.nn as nn
import torch.nn.init as init
 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import optuna

# SEED FOR REPRODUCIBILITY
# so we get the same results every time we run the notebook
SEED = 42

# !!!
# Set the random seed for both PyTorch and NumPy to ensure reproducibility of results.
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check if an NVIDIA GPU (CUDA) is available. 
# If yes, we use 'cuda'; otherwise, we fall back to the 'cpu'.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device choice so we know if the code is running on the GPU or CPU
print(f"Using device: {DEVICE}\n")

# 0. DATASET LOADING -------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv("1assign/insurance.csv")
df.head()

#TODO: Check the dataset for any issues (missing values, data types, etc.) and preprocess as needed.
df.isnull().sum()

# -------------------------------------------------------------------------------------------------------------------------------------------------------- 

# 1.1 FEATURE ENGENEERING -------------------------------------------------------------------------------------------------------------------------------------
# We use get_dummies to transform the 'region' column into multiple numeric columns.
# If 'region' has 4 categories (NE, NW, SE, SW), it will create 4 columns.
# 'drop_first=True' is used to avoid the "Dummy Variable Trap":
# It removes one column (e.g., 'region_northeast'). 
# If the remaining columns (NW, SE, SW) are all 0, the model knows the region MUST be NE.
# This prevents mathematical redundancy (multicollinearity) in your model.
data = pd.get_dummies(df, columns=['region'], drop_first=True)
# We manually map the strings to integers: Female becomes 0, Male becomes 1.
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
# 'Binary Encoding' for smoking status:
# 'no' becomes 0 and 'yes' becomes 1.
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
# Display the first 5 rows of the new DataFrame to verify the changes.
data.head()
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 1.2 PREPROCESSING (NORMALIZATION, STANDARDIZATION, ETC.)
# Define which numeric columns need to be scaled. It is not necessary to scale the one-hot encoded 'region' columns because they are already in a 0/1 format.
# We include 'charges' here because it's our target variable and we want it on a similar scale as the features.
cols_a_escalar = ['age', 'bmi', 'children', 'charges']

# Initialize the StandardScaler.
# This follows the formula: z = (x - mean) / standard_deviation
# It centers the data around 0 with a standard deviation of 1.
scaler = StandardScaler()

# fit_transform does two things:
# 1. 'fit' calculates the average (mean) and variance of each column.
# 2. 'transform' actually applies the math to change the numbers.
data[cols_a_escalar] = scaler.fit_transform(data[cols_a_escalar])
# Ensure all data in the DataFrame is a float (decimal number).
# This prevents errors during PyTorch tensor conversion later.
data = data.astype(float)

# X = every column except 'charges' (independent variable)
X = data.drop(columns=['charges'])
# y = only the column 'charges' (dependent variable, what we want to predict)
y = data['charges']

data.head()
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 1.3 DATA SPLITTING (TRAIN/VAL/TEST)

# We split the data: 80% for training and 20% for testing.
# random_state=42 ensures the "random" split is the same every time you run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We print the number of rows in each set 
print(f"Muestras de entrenamiento: {len(X_train)}")
print(f"Muestras de prueba: {len(X_test)}")
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 2.2 ACTIVATION FUNCTIONS
# Instanciamos el modelo definiendo las activaciones dentro
class InsuranceModel(nn.Module):
    def __init__(self, input_dim):
        super(InsuranceModel, self).__init__() # initializes the pytorch functionality for our model
        # defines the path of the data through the layers and activations in a single sequential block.
        self.net = nn.Sequential(
            # First layer: Multiplies the input features by weights and adds a bias to produce 64 new features.
            nn.Linear(input_dim, 64),
            # Applies the ReLU activation function to introduce non-linearity.
            nn.ReLU(),             
            # Second layer: Takes the 64 features from the first layer and transforms them into 32 new features.
            nn.Linear(64, 32),
            # Second ReLU activation to add non-linearity after the second layer.
            nn.ReLU(),   
            # Output layer: Takes the 32 features from the second layer and produces a single output (the predicted insurance cost).  
            # No activation function here because it's a regression problem; we want the output to be able to take any value, not just positive integers.         
            nn.Linear(32, 1)    
        )
        
    def forward(self, x):
        # The forward method defines how the input data flows through the network.
        return self.net(x)

#IMPLEMENTATION OF OPTUNA
class OptunaInsuranceModel(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(OptunaInsuranceModel, self).__init__()
        layers = []
        in_features = input_dim
        
        # Build the hidden layers based on Optuna's suggestion
        for h_size in hidden_layers:
            layers.append(nn.Linear(in_features, h_size))
            layers.append(nn.ReLU())
            in_features = h_size
            
        # Final output layer
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 3.1 TENSOR INITIALIZATION
# AFTER splitting and scaling
X_train_t = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test_t = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 3.1 LOSS FUNCTION (Regression --> MSE?)
criterion = nn.MSELoss()
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 3.2 INITIALIZATION (He Initialization)
# Obtains the number of input features from X_train (the number of columns) to initialize the model correctly.
input_size = X_train.shape[1]
# We create an instance of the InsuranceModel class, passing the number of input features as an argument.
model = InsuranceModel(input_size)

# He initialization (also known as Kaiming initialization) is a method for initializing the weights of neural networks, 
# especially those using ReLU activations.
def init_weights(m):
    # We check if the layer 'm' is an instance of nn.Linear, which means it's a fully connected layer.
    # We don't want to initialize activation functions or other types of layers,
    # only the linear layers that have weights and biases.
    if isinstance(m, nn.Linear):
        # We use Kaiming normal initialization for the weights of the layer 'm'.
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        # We set the bias of the layer 'm' to a small constant value (0.01) to ensure that the 
        # neurons are active at the beginning of training.
        m.bias.data.fill_(0.01)

# Aplicarlo a tu modelo
model.apply(init_weights)   

print(model)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 3.3 OPTIMIZER (SGD, ADAM, ..., ?)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr=1e-3 or lr=0.001?
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 4 MODEL TRAINING
def train(model, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()        # clear old gradients first
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

train(model, optimizer, epochs=100)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 5 PERFORMANCE EVALUATION
model.eval()
with torch.no_grad():
    preds = model(X_test_t).numpy()

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 6 HYPERPARAMETER TUNING WITH OPTUNA
def objective(trial):
    # 1. SUGGEST NUMBER OF LAYERS (e.g., between 1 and 3)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    hidden_layers = []
    for i in range(n_layers):
        # Suggest neurons for each layer (e.g., layer_0, layer_1...)
        size = trial.suggest_int(f"n_units_l{i}", 16, 128, step=16)
        hidden_layers.append(size)
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # 2. BUILD AND TRAIN
    model = OptunaInsuranceModel(X_train.shape[1], hidden_layers).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t.to(DEVICE))
        loss = criterion(preds, y_train_t.to(DEVICE))
        loss.backward()
        optimizer.step()

    # 3. EVALUATE
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t.to(DEVICE))
        score = r2_score(y_test_t.cpu(), test_preds.cpu())
        
    return score

# Creamos el estudio
# Busca la línea donde creas el estudio y cámbiala por esta:
study = optuna.create_study(direction="maximize") 


# Ejecutamos 30 o 50 intentos para tener datos suficientes
study.optimize(objective, n_trials=500)

# 1. Obtener el mejor R2 (el valor más alto)
mejor_r2 = study.best_value

# 2. Obtener los parámetros exactos que lograron ese R2
mejores_params = study.best_params

# RESULTS OF SIMPLE MODEL
print("RESULTS OF SIMPLE MODEL:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.3f}")

# RESULTS OF OPTUNA TUNING
print("\nRESULTS OF OPTUNA TUNING:")
print("="*30)
print(f"EL MEJOR VALOR DIRECTO: {mejor_r2:.4f}")
print("="*30)
print("Configuración ganadora:")
for parametro, valor in mejores_params.items():
    print(f" -> {parametro}: {valor}")

print(f"Total de intentos guardados: {len(study.trials)}")