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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

#Check the dataset for any issues (missing values, data types, etc.) and preprocess as needed.
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

# We apply it to the model 
model.apply(init_weights)   

print(model)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 3.3 OPTIMIZER (SGD, ADAM, ..., ?)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# lr = learning rate, which controls how much we adjust the weights in response to the calculated error each time the model weights are updated.
# lr = 0.001 is a common starting point for Adam (usually provides the best stability), but it can be tuned later with Optuna.
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 4 MODEL TRAINING
def train(model, optimizer, epochs=100):
    # an "epoch" is one complete pass through the entire training dataset.
    for epoch in range(epochs):

        # sets the model to training mode. This is important because some layers (like dropout or batch normalization) behave differently during training and evaluation.
        model.train()
        # 1. GRADIENT CLEANUP: PyTorch accumulates gradients by default. 
        # We must clear the old gradients from the previous step so they 
        # don't interfere with the current calculation.
        optimizer.zero_grad()  
        # 2. FORWARD PASS: Pass the input features (X) through the model.
        # The model performs matrix multiplications and activations to produce predictions. 
        preds = model(X_train_t)
        # 3. LOSS CALCULATION: Compare the model's predictions to the actual target values.
        # 'loss' is a single number representing how "wrong" the model is.
        loss = criterion(preds, y_train_t)
        # 4. BACKPROPAGATION (Backward Pass): PyTorch calculates the derivative 
        # of the loss with respect to every weight in the model. 
        # It finds which direction to move the weights to reduce error.
        loss.backward()
        # 5. OPTIMIZATION STEP: The optimizer (Adam) updates the weights 
        # based on the gradients calculated in the previous step and the Learning Rate.
        optimizer.step()
        # Every 10 epochs, print the current loss to ensure 
        # the model is actually converging (the number should be decreasing).
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
# Execute the training function to train the model for 100 epochs.
train(model, optimizer, epochs=100)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 5 PERFORMANCE EVALUATION

# 1. EVALUATION MODE: This tells the model to stop training. 
# It disables layers like Dropout or Batch Normalization so that 
# the model behaves consistently during testing.
model.eval()
# 2. DISABLE GRADIENT CALCULATION: Since we are only predicting 
# (not updating weights), we don't need to track gradients. 
# This saves a lot of memory and makes the computation much faster.
with torch.no_grad():
    # We pass the test data through the model and convert the 
    # resulting PyTorch tensor back into a NumPy array for Scikit-Learn.
    preds = model(X_test_t).numpy()

# 3. METRIC CALCULATION:
# RMSE (Root Mean Squared Error): Penalizes large errors more heavily. 
# It tells you, on average, how many units the predictions are off.
rmse = np.sqrt(mean_squared_error(y_test, preds))
# MAE (Mean Absolute Error): The average of the absolute differences 
# between predictions and actual values.
mae = mean_absolute_error(y_test, preds)
# R2 Score (Coefficient of Determination): Tells how much of the 
# variance in insurance charges is explained by the model. 
# 1.0 is a perfect fit; 0.0 means the model is no better than guessing the average.
r2 = r2_score(y_test, preds)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 6 HYPERPARAMETER TUNING WITH OPTUNA
def objective(trial):
    # Optuna suggests how many hidden layers to create (1 to 3).
    # Considering more layers could generate overfitting, as we are working with a relatively small data set.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    hidden_layers = []
    for i in range(n_layers):
        # For each layer, it suggests a number of neurons (16, 32, ..., 128).
        size = trial.suggest_int(f"n_units_l{i}", 16, 128, step=16)
        # step=16: limits the choices to 8 jumps, which reduces the search space and speeds up the optimization process.
        hidden_layers.append(size)
    # suggests values for the learning rate (lr) between 0.0001 and 0.01 on a logarithmic scale -->
    # log = True means that the search will be more efficient because it will explore more values in the lower range (where good learning rates often are) 
    # and fewer values in the higher range.
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Create the model using the suggested parameters. 
    # .to(DEVICE) moves the model to the GPU if available, which can significantly speed up training.
    model = OptunaInsuranceModel(X_train.shape[1], hidden_layers).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Suggest the number of epochs
    # We test between 50 (fast) and 500 (thorough)
    epochs = trial.suggest_int("epochs", 50, 500)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t.to(DEVICE))
        loss = criterion(preds, y_train_t.to(DEVICE))
        loss.backward()
        optimizer.step()

    # EVALUATE
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t.to(DEVICE))
        score = r2_score(y_test_t.cpu(), test_preds.cpu())
        
    return score

# Create the "Study" and tell it to maximize the R2 Score (higher is better).
study = optuna.create_study(direction="maximize") 

# Run 500 different trials to find the winning combination.
study.optimize(objective, n_trials=500)

# Extract the winning results
mejor_r2 = study.best_value
mejores_params = study.best_params
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# 7 FINAL MODEL TRAINING AND EVALUATION WITH THE WINNING CONFIGURATION
best_layers = study.best_params['n_layers']
best_lr = study.best_params['lr']
best_epochs = study.best_params['epochs']

best_hidden_layers = [study.best_params[f"n_units_l{i}"] for i in range(best_layers)]

final_model = OptunaInsuranceModel(input_dim=X_train.shape[1], hidden_layers=best_hidden_layers).to(DEVICE)
final_model.apply(init_weights)

# 2. Setup the winning Optimizer and Criterion
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr)
criterion = nn.MSELoss()

for epoch in range(best_epochs):
    final_model.train()
    optimizer.zero_grad()
    preds = final_model(X_train_t.to(DEVICE))
    loss = criterion(preds, y_train_t.to(DEVICE))
    loss.backward()
    optimizer.step()

# 4. Final Evaluation on the Test Set
final_model.eval()
with torch.no_grad():
    final_preds_t = final_model(X_test_t.to(DEVICE))
    # Move back to CPU and convert to numpy for sklearn metrics
    final_preds = final_preds_t.cpu().numpy()

# 5. Calculate the Missing Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

final_rmse = np.sqrt(mean_squared_error(y_test_t.cpu(), final_preds))
final_mae = mean_absolute_error(y_test_t.cpu(), final_preds)
final_r2 = r2_score(y_test_t.cpu(), final_preds)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# RESULTS OF SIMPLE MODEL
print("="*30)
print("METRICS OF SIMPLE MODEL:")
print("="*30)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.3f}")

# RESULTS OF OPTUNA TUNING
print("="*30)
print("RESULTS OF OPTUNA TUNING:")
print("="*30)
print(f"Best R2 Score: {mejor_r2:.4f}")
print("Winning Configuration:")
for parametro, valor in mejores_params.items():
    print(f" -> {parametro}: {valor}")
print("METRICS: ")
print(f"RMSE: {final_rmse:.4f}")
print(f"MAE:  {final_mae:.4f}")
print(f"R2:   {final_r2:.4f}")


#--------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. VISUALIZATIONS

def to_numpy(x):
    if hasattr(x, "detach"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.array(x)
    return arr.reshape(-1)

y_true   = to_numpy(y_test)
y_simple = to_numpy(preds)[:len(y_true)]
y_optuna = to_numpy(final_preds)[:len(y_true)]

print(f"Shapes — y_true: {y_true.shape}, y_simple: {y_simple.shape}, y_optuna: {y_optuna.shape}")

residuals_simple = y_true - y_simple
residuals_optuna = y_true - y_optuna

# Layout
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Simple Model vs Optuna-Tuned Model", fontsize=16, fontweight="bold", y=0.98)

BLUE   = "#378ADD"   # simple model
CORAL  = "#D85A30"   # optuna model
GRAY   = "#B4B2A9"
LIGHT  = "#F1EFE8"

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

#  1. Predicted vs Actual
ax1 = fig.add_subplot(gs[0, 0])

lim = (min(y_true.min(), y_simple.min(), y_optuna.min()) - 0.1,
       max(y_true.max(), y_simple.max(), y_optuna.max()) + 0.1)

ax1.scatter(y_true, y_simple, alpha=0.45, s=25, color=BLUE,  label=f"Simple  (R²={r2:.3f})", zorder=3)
ax1.scatter(y_true, y_optuna, alpha=0.45, s=25, color=CORAL, label=f"Optuna  (R²={final_r2:.3f})", zorder=3)
ax1.plot(lim, lim, "--", color=GRAY, linewidth=1.2, label="Perfect fit", zorder=2)
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("Actual (standardised charges)")
ax1.set_ylabel("Predicted")
ax1.set_title("Predicted vs Actual")
ax1.legend(fontsize=8)
ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

# 2. Residuals distribution
ax2 = fig.add_subplot(gs[0, 1])

bins = np.linspace(min(residuals_simple.min(), residuals_optuna.min()),
                   max(residuals_simple.max(), residuals_optuna.max()), 40)

ax2.hist(residuals_simple, bins=bins, alpha=0.55, color=BLUE,  label="Simple", edgecolor="white", linewidth=0.4)
ax2.hist(residuals_optuna, bins=bins, alpha=0.55, color=CORAL, label="Optuna", edgecolor="white", linewidth=0.4)
ax2.axvline(0, color=GRAY, linestyle="--", linewidth=1.2)
ax2.set_xlabel("Residual (actual − predicted)")
ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution")
ax2.legend(fontsize=8)
ax2.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

# 3. Metrics bar chart
ax3 = fig.add_subplot(gs[1, 0])

metrics      = ["RMSE", "MAE"]
simple_vals  = [rmse, mae]
optuna_vals  = [final_rmse, final_mae]

x     = np.arange(len(metrics))
width = 0.32

bars_s = ax3.bar(x - width / 2, simple_vals, width, label="Simple", color=BLUE,  alpha=0.85, edgecolor="white", linewidth=0.6)
bars_o = ax3.bar(x + width / 2, optuna_vals, width, label="Optuna", color=CORAL, alpha=0.85, edgecolor="white", linewidth=0.6)

for bar in list(bars_s) + list(bars_o):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.004,
             f"{bar.get_height():.3f}",
             ha="center", va="bottom", fontsize=8)

ax3.set_xticks(x); ax3.set_xticklabels(metrics)
ax3.set_ylabel("Error (standardised scale)")
ax3.set_title("RMSE & MAE Comparison\n(lower is better)")
ax3.legend(fontsize=8)
ax3.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)

# 4. R² comparison 
ax4 = fig.add_subplot(gs[1, 1])

model_names = ["Simple", "Optuna"]
r2_vals     = [r2, final_r2]
colors      = [BLUE, CORAL]

bars = ax4.barh(model_names, r2_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.6, height=0.4)
ax4.set_xlim(0, 1.08)
ax4.axvline(1, color=GRAY, linestyle="--", linewidth=1.0, label="Perfect R²=1")

for bar, val in zip(bars, r2_vals):
    ax4.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

ax4.set_xlabel("R² Score")
ax4.set_title("R² Score Comparison\n(higher is better)")
ax4.legend(fontsize=8)
ax4.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.6)

# Show
plt.show()