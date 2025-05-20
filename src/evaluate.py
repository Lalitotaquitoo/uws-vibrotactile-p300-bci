import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from preprocessing import create_epochs


def evaluate(mat_path, model_dir):
    model = load_model(os.path.join(model_dir, 'eegnet_model_final.h5'))
    mean = np.load(os.path.join(model_dir, 'X_mean.npy'))
    std = np.load(os.path.join(model_dir, 'X_std.npy'))

    epochs = create_epochs(mat_path)
    X = epochs.get_data()
    X = (X - mean) / std
    X = X[..., np.newaxis]

    y_raw = epochs.events[:, 2]
    label_map = {1:0, 2:1}
    y = np.array([label_map[v] for v in y_raw if v in label_map])
    y_cat = to_categorical(y, num_classes=2)

    loss, acc = model.evaluate(X, y_cat, verbose=1)
    print(f"Evaluation Accuracy: {acc:.4f}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Path to .mat file to evaluate')
    p.add_argument('--model-dir', default='models', help='Directory with saved model')
    args = p.parse_args()
    evaluate(args.input, args.model_dir)