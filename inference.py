import torch
from kan import KAN
import data_utils
import model_utils
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_deployed_model(checkpoint_dir="./model"):
    # The KAN library creates checkpoints with _state and _cache_data
    # We need to initialize the model first
    model = KAN(width=[7, 1], grid=10, k=3, device="cpu")
    
    # Load state manually
    try:
        # Based on library source: torch.save(model.state_dict(), f'{path}_state')
        # And torch.save(model.cache_data, f'{path}_cache_data')
        
        path = f"{checkpoint_dir}/0.0" # Based on the generated files in ./model/
        
        state_dict = torch.load(f'{path}_state', map_location='cpu')
        model.load_state_dict(state_dict)
        
        model.cache_data = torch.load(f'{path}_cache_data', map_location='cpu')
        
        print(f"Model berhasil dimuat dari {checkpoint_dir}")
        return model
    except Exception as e:
        print(f"Error memuat model: {e}")
        return None

def run_inference():
    # 1. Fetch data
    print("Fetching latest data for inference...")
    df = data_utils.fetch_data()
    df = data_utils.compute_bollinger(df)
    
    # 2. Preprocess (need to ensure features match training)
    # Note: In production, normalization stats (X_mean, X_std) 
    # should be saved during training and reloaded here!
    
    _, _, df_ready = data_utils.create_dataset(df)

    # print(df_ready.head().to_string())
    
    # Ambil data terbaru (terakhir)
    latest_data = df_ready.iloc[[-1]]
    features = ['bb_pct_scaled', 'return_lag_1', 'return_lag_2', 
                'rolling_mean_return_5', 'rolling_vol_5', 'ratio_to_1w', 'ratio_to_1m']
    
    X = torch.tensor(latest_data[features].values, dtype=torch.float32)
    
    # 3. Load Model & Stats
    model = load_deployed_model()
    X_mean, X_std = model_utils.load_normalization_stats()
    
    if model is None:
        return

    # 4. Predict
    # Apply normalization using saved stats
    X_norm = model_utils.normalize_data(X, X_mean, X_std)
    
    with torch.no_grad():
        # Using sigmoid to get probability
        pred = torch.sigmoid(model(X_norm)).item()
    
    status = "Buy" if pred > 0.6 else "Sell" if pred < 0.4 else "Hold"
    
    # Calculate SL/TP levels based on Bollinger Bands
    price = latest_data['close_1d'].item()
    upper = latest_data['bb_upper'].item()
    lower = latest_data['bb_lower'].item()
    
    print(f"\nPrediksi Terbaru: Probabilitas Buy = {pred:.4f} -> {status}")
    
    if status != "Hold":
        entry = price
        sl = lower if status == "Buy" else upper
        tp = upper if status == "Buy" else lower
        
        print(f"--- Trading Plan ---")
        print(f"Entry      : {entry:.2f}")
        print(f"Stop Loss  : {sl:.2f}")
        print(f"Take Profit: {tp:.2f}")

if __name__ == "__main__":
    run_inference()
