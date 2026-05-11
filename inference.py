import torch
import pandas as pd
import numpy as np
from kan import KAN
from pytorch_forecasting import TemporalFusionTransformer
from main import download_mtf_data, prepare_features
from sklearn.preprocessing import StandardScaler

def run_inference(ticker="PTBA.JK"):
    # 1. Load Data
    print(f"Loading fresh data for {ticker}...")
    d15, d1h, dd = download_mtf_data(ticker)
    df = prepare_features(d15, d1h, dd)
    
    features = ['log_return', 'smoothed_log', 'log_return_h1', 'log_return_d1', 'z_score_sr']
    X = df[features].values
    
    # 2. Load KAN Model
    print("Loading KAN model...")
    # Initialize with same width as training, auto_save=False to avoid unwanted folder creation
    kan_model = KAN(width=[len(features), 8, 3], grid=3, k=3, auto_save=False)
    kan_model.loadckpt(path="./model/kan_model.ckpt")
    
    # KAN Prediction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        kan_preds = kan_model(X_tensor)
        kan_classes = torch.argmax(kan_preds, dim=1).numpy()
    
    # 3. Load TFT model
    print("Loading TFT model...")
    tft_model = TemporalFusionTransformer.load_from_checkpoint("./model/tft_model.ckpt", map_location="cpu")
    tft_model.eval()
    
    # TFT Prediction - Use TimeSeriesDataSet for correct formatting
    from pytorch_forecasting import TimeSeriesDataSet
    
    # Use the same parameters as training
    max_encoder_length = 24
    # We need a dummy target for the dataset creation, even if not used for prediction
    df['target'] = 0 
    
    try:
        # Load saved dataset parameters
        import pickle
        with open("./model/dataset_params.pkl", "rb") as f:
            ds_params = pickle.load(f)
        
        # Manually create the dataset using the parameters
        inference_ds = TimeSeriesDataSet(
            df,
            **ds_params
        )
        inf_loader = inference_ds.to_dataloader(train=False, batch_size=len(df), num_workers=0)
        
        batch = next(iter(inf_loader))
        x, _ = batch
        
        with torch.no_grad():
            output = tft_model(x)
            # output is likely a namedtuple or dict depending on version
            # Usually .prediction or ['prediction']
            if hasattr(output, 'prediction'):
                logits = output.prediction
            else:
                logits = output
                
            # If logits has shape (Batch, Horizon, Classes), take last horizon (which is 1)
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]
            
            tft_classes = torch.argmax(logits, dim=-1).numpy()
            
            # Pad with -1 for points where we didn't have enough encoder history
            padding_needed = len(df) - len(tft_classes)
            if padding_needed > 0:
                tft_classes = np.pad(tft_classes, (padding_needed, 0), constant_values=-1)
            elif padding_needed < 0:
                tft_classes = tft_classes[-len(df):]
                
    except Exception as e:
        print(f"TFT Prediction failed: {e}")
        tft_classes = np.zeros(len(df)) - 1
    
    # 4. Comparative Output
    # Align lengths (TFT might have different length due to encoder_length)
    # For simplicity, we just look at the last few points
    results = pd.DataFrame({
        'Timestamp': df.index,
        'Price': df['Close'],
        'KAN_Signal': kan_classes,
        'TFT_Signal': tft_classes[-len(df):] if len(tft_classes) >= len(df) else np.pad(tft_classes, (len(df)-len(tft_classes), 0), constant_values=-1)
    })
    
    # Signal mapping: 0: Sell/Loss, 1: Hold/Timeout, 2: Buy/Profit
    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY", -1: "N/A"}
    results['KAN_Action'] = results['KAN_Signal'].map(signal_map)
    results['TFT_Action'] = results['TFT_Signal'].map(signal_map)
    
    print("\nRecent Comparative Inference Results:")
    print(results[['Timestamp', 'Price', 'KAN_Action', 'TFT_Action']].tail(10))

if __name__ == "__main__":
    run_inference(ticker="JPFA.JK")
