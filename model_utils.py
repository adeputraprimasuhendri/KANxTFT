import torch
import json
from kan import KAN

def save_normalization_stats(X_mean, X_std, filepath="./model/norm_stats.json"):
    stats = {
        "mean": X_mean.tolist(),
        "std": X_std.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(stats, f)

def load_normalization_stats(filepath="./model/norm_stats.json"):
    with open(filepath, 'r') as f:
        stats = json.load(f)
    return torch.tensor(stats["mean"]), torch.tensor(stats["std"])

def get_normalization_stats(X):
    X_mean = torch.mean(X, dim=0)
    X_std  = torch.std(X, dim=0)
    return X_mean, X_std

def normalize_data(X, X_mean, X_std):
    return (X - X_mean) / (X_std + 1e-6)

def prepare_model(X_train):
    input_dim = X_train.shape[1]
    X_mean, X_std = get_normalization_stats(X_train)
    X_train_norm = normalize_data(X_train, X_mean, X_std)

    model = KAN(width=[input_dim, 1], grid=10, k=3, device="cpu")
    model.update_grid_from_samples(X_train_norm)
    return model, X_mean, X_std


def train_model(model, X_norm, Y, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_norm)
        loss = loss_fn(y_pred, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def get_predictions(model, X_norm):
    with torch.no_grad():
        y_pred_all = torch.sigmoid(model(X_norm)).numpy()
    return y_pred_all

def evaluate_model(y_true, y_pred_prob):
    # Convert continuous values to discrete classes
    # 1.0 -> Buy, 0.5 -> Hold, 0.0 -> Sell
    def to_class(val):
        if val > 0.6: return 1.0  # Buy
        if val < 0.4: return 0.0  # Sell
        return 0.5                # Hold

    y_true_cls = [to_class(v) for v in y_true.flatten()]
    y_pred_cls = [to_class(v) for v in y_pred_prob.flatten()]
    
    classes = [1.0, 0.5, 0.0]
    class_names = {1.0: "Buy ", 0.5: "Hold", 0.0: "Sell"}
    
    results = {}
    total_correct = 0
    
    print("\n--- Evaluasi Metrik Klasifikasi ---")
    print(f"{'Class':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 45)
    
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        total_correct += tp
        print(f"{class_names[cls]:<6} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
        
    accuracy = total_correct / len(y_true_cls)
    print("-" * 45)
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    
    # Return metrics for plotting
    final_metrics = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        final_metrics[class_names[cls].strip()] = {"Precision": precision, "Recall": recall, "F1-Score": f1}
    
    return accuracy, final_metrics
