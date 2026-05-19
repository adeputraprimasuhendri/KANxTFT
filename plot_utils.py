import matplotlib.pyplot as plt
import numpy as np

def plot_results(close_1d, bb_upper, bb_mid, bb_lower, Y, y_pred_all, min_len, train_size, last_n=30):
    plt.figure(figsize=(12, 6))

    # Filter for last N days
    start_idx = max(0, min_len - 1 - last_n)

    plt.plot(range(start_idx, min_len-1), close_1d[start_idx:min_len-1], label="Harga Gold Futures (1D)", color="blue", lw=1.0)
    plt.plot(range(start_idx, min_len-1), bb_upper[start_idx:min_len-1], color="gray",  lw=0.8, linestyle="--", label="BB Upper")
    plt.plot(range(start_idx, min_len-1), bb_mid[start_idx:min_len-1],   color="orange", lw=0.8, linestyle="--", label="BB Mid (SMA20)")
    plt.plot(range(start_idx, min_len-1), bb_lower[start_idx:min_len-1], color="gray",  lw=0.8, linestyle="--", label="BB Lower")

    # Vertical line for Train/Test split if within view
    if train_size >= start_idx:
        plt.axvline(x=train_size, color="black", linestyle="-", lw=2, label="Train/Test Split")

    y_numpy = Y.numpy()

    # Filter signals within view
    buy_signals  = [i for i, y in enumerate(y_numpy) if y == 1.0 and i >= start_idx]
    sell_signals = [i for i, y in enumerate(y_numpy) if y == 0.0 and i >= start_idx]

    plt.scatter(buy_signals,  close_1d[buy_signals],  marker="^", color="green", label="Buy Signal (Target)", zorder=5, alpha=0.5)
    plt.scatter(sell_signals, close_1d[sell_signals], marker="v", color="red",   label="Sell Signal (Target)", zorder=5, alpha=0.5)

    pred_buy  = [i for i, p in enumerate(y_pred_all) if p > 0.6 and i >= start_idx]
    pred_sell = [i for i, p in enumerate(y_pred_all) if p < 0.4 and i >= start_idx]

    plt.scatter(pred_buy,  close_1d[pred_buy],  marker=".", color="lime", label="Model Buy", zorder=4)
    plt.scatter(pred_sell, close_1d[pred_sell], marker=".", color="magenta", label="Model Sell", zorder=4)

    plt.title(f"Buy/Sell Signals + Bollinger Bands (Last {last_n} Days)")
    plt.ylabel("Harga")
    plt.xlabel("Sample")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    print(f"Plot prediksi disimpan ke: prediction_plot.png (Zoomed last {last_n} days)")


def plot_metrics(train_metrics, test_metrics):
    labels = ['Precision', 'Recall', 'F1-Score']
    classes = ['Buy', 'Hold', 'Sell']

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, cls in enumerate(classes):
        train_vals = [train_metrics[cls][l] for l in labels]
        test_vals  = [test_metrics[cls][l] for l in labels]

        x = np.arange(len(labels))
        width = 0.35

        axs[i].bar(x - width/2, train_vals, width, label='Train', color='skyblue')
        axs[i].bar(x + width/2, test_vals, width, label='Test', color='salmon')

        axs[i].set_title(f'Metrics for {cls}')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels)
        axs[i].set_ylim(0, 1.1)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    print("Plot metrik evaluasi disimpan ke: metrics_plot.png")


def plot_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color="purple")
    plt.title("Loss selama Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")
    print("Plot loss disimpan ke: loss_plot.png")
