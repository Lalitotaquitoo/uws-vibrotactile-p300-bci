import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample, shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from preprocessing import create_epochs
from eegnet import EEGNet


def load_and_preprocess(mat_path):
    epochs = create_epochs(mat_path)
    X = epochs.get_data()  # shape: (n_epochs, n_chans, n_times)
    y_raw = epochs.events[:, 2]
    label_map = {1:0, 2:1}
    y = np.array([label_map[v] for v in y_raw if v in label_map])
    X = X[:len(y)]

    # Normalize
    mean = X.mean(axis=(0,2), keepdims=True)
    std = X.std(axis=(0,2), keepdims=True)
    X_norm = (X - mean) / std
    X_norm = X_norm[..., np.newaxis]
    return X_norm, y, mean, std


def main(data_dir, model_dir):
    # Example with one subject file
    mat_file = os.path.join(data_dir, 'P1_high1.mat')
    X, y, mean, std = load_and_preprocess(mat_file)

    # Balance classes
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    Xb, yb = [], []
    for cls in classes:
        Xc = X[y==cls]; yc = y[y==cls]
        Xr, yr = resample(Xc, yc, replace=True, n_samples=min_count, random_state=42)
        Xb.append(Xr); yb.append(yr)
    Xb = np.concatenate(Xb); yb = np.concatenate(yb)
    Xb, yb = shuffle(Xb, yb, random_state=42)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(Xb, yb), 1):
        X_train, X_val = Xb[train_idx], Xb[val_idx]
        y_train, y_val = yb[train_idx], yb[val_idx]
        y_train_cat = to_categorical(y_train, 2)
        y_val_cat = to_categorical(y_val, 2)

        model = EEGNet(2, X.shape[1], X.shape[2])
        model.compile('adam', 'categorical_crossentropy', ['accuracy'])

        callbacks = [
            EarlyStopping('val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau('val_loss', factor=0.5, patience=5)
        ]

        model.fit(X_train, y_train_cat,
                  validation_data=(X_val, y_val_cat),
                  epochs=50, batch_size=32,
                  callbacks=callbacks, verbose=1)

        loss, acc = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        accs.append(acc)

    # Save final model and normalization
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'eegnet_model_final.h5'))
    np.save(os.path.join(model_dir, 'X_mean.npy'), mean)
    np.save(os.path.join(model_dir, 'X_std.npy'), std)
    print(f"Average CV Accuracy: {np.mean(accs):.4f}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data', help='Directory with .mat files')
    p.add_argument('--model-dir', default='models', help='Output directory for models')
    args = p.parse_args()
    main(args.data_dir, args.model_dir)