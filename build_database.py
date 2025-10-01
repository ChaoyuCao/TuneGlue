import torch
import joblib
import json
import numpy as np
from train import NeuralNetwork


def json_saver(path, data):
        data_serializable = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }
        with open(path, 'w') as json_file:
            json.dump(data_serializable, json_file)


def predict_ensemble(model_type, X_input, target='deformation', num_folds=5):
    predictions = []
    
    # Load the scaler
    scaler = joblib.load(f'models/{target}/scaler.pkl')
    X_input_scaled = scaler.transform(X_input)
    
    if model_type == 'NN':
        for fold in range(1, num_folds + 1):
            # Load the entire NN model
            model = torch.load(f'models/{target}/NN_fold{fold}.pth')
            model.eval()
            X_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)
            with torch.no_grad():
                pred = model(X_tensor).numpy().flatten()
            predictions.append(pred)
    else:
        for fold in range(1, num_folds + 1):
            # Load the entire scikit-learn model
            model = joblib.load(f'models/{target}/{model_type}_fold{fold}.pkl')
            pred = model.predict(X_input_scaled)
            predictions.append(pred)
    
    # Return mean predictions
    return np.mean(predictions, axis=0)
    
    
def build_inputs(N):
    x = np.random.rand(N, 4)
    x = x / np.sum(x, axis=-1)[:, None]
    x_schc = np.random.rand(N, 2)
    x_schc = x_schc / np.sum(x_schc, axis=-1)[:, None]
    x = np.concatenate([x, x_schc], axis=-1)
    return x


if __name__ == '__main__':
    x = build_inputs(100000)
    modulus = predict_ensemble('RFR', x[:, :-1], target='modulus')
    strength = predict_ensemble('NN', x[:, :-1], target='strength')
    deformation = predict_ensemble('NN', x[:, :-1], target='deformation') * 800.0
    y = np.concatenate([modulus[:, None], strength[:, None], deformation[:, None]], axis=-1)
    database = {'formula': x, 'tissue': y}
    json_saver('database/database.json', database)

