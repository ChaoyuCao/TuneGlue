import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os


def load_data(file_path, target):
    if target == 'deformation':
        df = pd.read_csv(file_path).clip(upper=800)
    else: 
        df = pd.read_csv(file_path)
    df = df.drop('HC', axis=1)
    X = df.iloc[:, :5].values / 100.0
    
    indices = {'modulus': -3, 'strength': -2, 'deformation': -1}
    y = df.iloc[:, indices[target]].values 
    if target == 'deformation':
        y = y / 800.0
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    os.makedirs(f'models/{target}', exist_ok=True)
    joblib.dump(scaler, f'models/{target}/scaler.pkl')
    return X, y


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=5, layer_sizes=[16, 8], output_size=1, dropout=0.1, sigmoid=False):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        self.dropout = nn.Dropout(dropout)
        
        if layer_sizes is not None:
            for size in layer_sizes:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                layers.append(self.dropout)
                prev_size = size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        self.sigmoid = sigmoid
    
    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.network(x))
        else:
            return self.network(x)


def train_model(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.005, weight_decay=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    best_model_state = None
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    
    # Mini-batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            # Add noise augmentation
            noise = torch.randn_like(batch_x) * 0.05  # 5% noise
            batch_x_aug = batch_x + noise
            
            optimizer.zero_grad()
            outputs = model(batch_x_aug)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return best_model_state, best_val_loss


def evaluate_model_nn(model, X_data, y_data):
    model.eval()
    X_data = torch.tensor(X_data, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_data).numpy().flatten()
        mse = mean_squared_error(y_data, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_data, predictions)
    return mse, rmse, r2


def main(seed, X, y, epochs, layer_sizes, lr, dropout, n_estimators, alpha, target):
    os.makedirs(f'models/{target}', exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    X_train_val, _, y_train_val, _ = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=seed)  # Val unused since CV

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    test_metrics = {
        'NN': {'rmse': [], 'r2': []},
        'RFR': {'rmse': [], 'r2': []},
        'XGB': {'rmse': [], 'r2': []},
        'SVR': {'rmse': [], 'r2': []},
        'Lasso': {'rmse': [], 'r2': []},
        'Ridge': {'rmse': [], 'r2': []},
        'ETR': {'rmse': [], 'r2': []}
    }

    sk_models = {
        'RFR': RandomForestRegressor(n_estimators=n_estimators, random_state=seed),
        'XGB': XGBRegressor(n_estimators=n_estimators, random_state=seed),
        'SVR': SVR(kernel='rbf'),
        'Lasso': Lasso(alpha=alpha),
        'Ridge': Ridge(alpha=alpha),
        'ETR': ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed)
    }
    
    # Evaluate scikit-learn models
    for name, sk_model in sk_models.items():
        print(f'\nEvaluating {name}')
        cv_rmse_scores = []
        cv_r2_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
            X_train_kf = X_train_val[train_idx]
            y_train_kf = y_train_val[train_idx]
            X_val_kf = X_train_val[val_idx]
            y_val_kf = y_train_val[val_idx]
            
            sk_model.fit(X_train_kf, y_train_kf)
            # Save
            joblib.dump(sk_model, f'models/{target}/{name}_fold{fold+1}.pkl')
            
            # Validation
            pred_val = sk_model.predict(X_val_kf)
            mse_val = mean_squared_error(y_val_kf, pred_val)
            rmse_val = np.sqrt(mse_val)
            r2_val = r2_score(y_val_kf, pred_val)
            
            # Test
            pred_test = sk_model.predict(X_test)
            mse_test = mean_squared_error(y_test, pred_test)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(y_test, pred_test)
            
            cv_rmse_scores.append(rmse_val)
            cv_r2_scores.append(r2_val)
            test_metrics[name]['rmse'].append(rmse_test)
            test_metrics[name]['r2'].append(r2_test)
            
            print(f'Fold {fold+1}: Val RMSE: {rmse_val:.4f}, Val R²: {r2_val:.4f}, Test RMSE: {rmse_test:.4f}, Test R²: {r2_test:.4f}')
        
        print(f'{name} Cross-Validation RMSE: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}')
        print(f'{name} Cross-Validation R²: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}')
        print(f'{name} Test RMSE across folds: {np.mean(test_metrics[name]["rmse"]):.4f} ± {np.std(test_metrics[name]["rmse"]):.4f}')
        print(f'{name} Test R² across folds: {np.mean(test_metrics[name]["r2"]):.4f} ± {np.std(test_metrics[name]["r2"]):.4f}')
        
    # Evaluate NN separately
    print('\nEvaluating NN')
    cv_rmse_scores = []
    cv_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
        X_train_kf = X_train_val[train_idx]
        y_train_kf = y_train_val[train_idx]
        X_val_kf = X_train_val[val_idx]
        y_val_kf = y_train_val[val_idx]
        
        model = NeuralNetwork(layer_sizes=layer_sizes, dropout=dropout, sigmoid=True if target == 'deformation' else False)  # Simplified architecture
        best_model_state, _ = train_model(model, X_train_kf, y_train_kf, X_val_kf, y_val_kf, 
                                          epochs=epochs, lr=lr, weight_decay=0.001)
        model.load_state_dict(best_model_state)
        torch.save(model, f'models/{target}/NN_fold{fold+1}.pth')
        # Validation
        mse_val, rmse_val, r2_val = evaluate_model_nn(model, X_val_kf, y_val_kf)
        
        # Test
        mse_test, rmse_test, r2_test = evaluate_model_nn(model, X_test, y_test)
        
        cv_rmse_scores.append(rmse_val)
        cv_r2_scores.append(r2_val)
        test_metrics['NN']['rmse'].append(rmse_test)
        test_metrics['NN']['r2'].append(r2_test)
        
        print(f'Fold {fold+1}: Val RMSE: {rmse_val:.4f}, Val R²: {r2_val:.4f}, Test RMSE: {rmse_test:.4f}, Test R²: {r2_test:.4f}')

    print(f'NN Cross-Validation RMSE: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}')
    print(f'NN Cross-Validation R²: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}')
    print(f'NN Test RMSE across folds: {np.mean(test_metrics["NN"]["rmse"]):.4f} ± {np.std(test_metrics["NN"]["rmse"]):.4f}')
    print(f'NN Test R² across folds: {np.mean(test_metrics["NN"]["r2"]):.4f} ± {np.std(test_metrics["NN"]["r2"]):.4f}')

    # Final summary of test set performance for all models
    print('\nFinal Test Set Performance (Mean Across Folds):')
    print('Model | RMSE | R²')
    print('-' * 30)
    for model_name in test_metrics:
        mean_rmse = np.mean(test_metrics[model_name]['rmse'])
        mean_r2 = np.mean(test_metrics[model_name]['r2'])
        print(f'{model_name:5} | {mean_rmse:6.4f} | {mean_r2:6.4f}')
        
        
if __name__ == '__main__':
    X, y = load_data('./data/data.csv', target='modulus')
    main(42, X, y, epochs=150, layer_sizes=[16, 8, 8], dropout=0.1, lr=5e-3, n_estimators=100, alpha=0.1, target='modulus')
    
    X, y = load_data('./data/data.csv', target='strength')
    main(11, X, y, epochs=150, layer_sizes=[32, 16, 8], dropout=0.3, lr=5e-3, n_estimators=100, alpha=0.1, target='strength')
    
    X, y = load_data('./data/data.csv', target='deformation')
    main(42, X, y, epochs=250, layer_sizes=[16, 8, 8], dropout=0.4, lr=5e-4, n_estimators=10, alpha=0.1, target='deformation')
    
    
    
    