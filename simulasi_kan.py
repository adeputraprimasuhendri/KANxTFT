import warnings
import data_utils
import model_utils
import plot_utils
import backtest_utils

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # 1. Fetch data
    print("Fetching data...")
    df = data_utils.fetch_data()

    # 2. Compute Indicators
    print("Computing indicators...")
    df = data_utils.compute_bollinger(df)

    # 3. Create Dataset
    print("Creating dataset and engineering features...")
    X, Y, df_ready = data_utils.create_dataset(df)

    # 4. Split Data (80% Train, 20% Test)
    print("Splitting data into Train and Test sets...")
    X_train, Y_train, X_test, Y_test = data_utils.split_dataset(X, Y, train_ratio=0.8)

    # 5. Prepare Model
    print("Preparing model...")
    model, X_mean, X_std = model_utils.prepare_model(X_train)
    model_utils.save_normalization_stats(X_mean, X_std) # Simpan stats
    X_train_norm = model_utils.normalize_data(X_train, X_mean, X_std)
    X_test_norm  = model_utils.normalize_data(X_test, X_mean, X_std)

    # 6. Train Model
    print("Training model...")
    losses = model_utils.train_model(model, X_train_norm, Y_train)
    print(f"Training selesai. Loss akhir: {losses[-1]:.6f}")

    # 7. Get Predictions
    print("Generating predictions...")
    y_pred_train = model_utils.get_predictions(model, X_train_norm)
    y_pred_test  = model_utils.get_predictions(model, X_test_norm)

    # 8. Evaluate Model
    print("\n--- EVALUASI DATA TRAINING ---")
    _, train_metrics = model_utils.evaluate_model(Y_train.numpy(), y_pred_train)

    print("\n--- EVALUASI DATA TESTING (UNSEEN) ---")
    _, test_metrics = model_utils.evaluate_model(Y_test.numpy(), y_pred_test)

    # 9. Show Latest Predictions (from Test set)
    print("\n5 Prediksi Terakhir di Data Test (Probabilitas Buy):")
    for i in range(-5, 0):
        target = Y_test[i].item()
        pred   = y_pred_test[i][0]
        status = "Buy" if pred > 0.6 else "Sell" if pred < 0.4 else "Hold"
        print(f"Sample {i}: Target={target:.1f}, Pred={pred:.4f} -> {status}")

    # 10. Backtesting
    print("\nRunning backtest on Test data...")
    train_size = len(X_train)
    df_test = df_ready.iloc[train_size:].copy()
    backtest_df, trade_df = backtest_utils.run_backtest_mt(df_test, y_pred_test.flatten())
    backtest_utils.plot_backtest(backtest_df)

    # 11. Visualization
    print("Saving plots...")
    import numpy as np
    y_pred_all = np.concatenate([y_pred_train, y_pred_test])
    plot_utils.plot_results(df_ready['close_1d'].values, df_ready['bb_upper'].values, 
                            df_ready['bb_mid'].values, df_ready['bb_lower'].values, 
                            Y, y_pred_all, len(df_ready)+1, len(X_train), last_n=30)
    plot_utils.plot_metrics(train_metrics, test_metrics)
    plot_utils.plot_loss(losses)

if __name__ == "__main__":
    main()
