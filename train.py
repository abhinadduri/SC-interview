import argparse
import torch
import wandb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from model import SCNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import save_curve_data, compute_metrics

parser = argparse.ArgumentParser(description="Cell Type Identification Training.")
parser.add_argument("--wandb-proj", default="Interview", help="Weights & Biases Project", dest="wandb_proj")
parser.add_argument("--group", default=None, help="Weights & Biases Group", dest="group")
parser.add_argument("--data", default="cells.npy", help="scRNA data to use for training", dest="data")
parser.add_argument("--seed", type=int, default=0, help="Random seed to use for train / val / test splits", dest="seed")
parser.add_argument("--lr", "--learning-rate", type=float, default=3e-4, help="initial learning rate", dest="lr",)
parser.add_argument("--embed-size", type=int, default=128, help="hidden dimension embedding size", dest="embed_size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train for", dest="epochs")
parser.add_argument("--num_hidden_layers", type=int, default=3, help="number of hidden layers to use in the model", dest="num_hidden_layers")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate", dest="dropout")
parser.add_argument("--recompute-train-loss", help="recompute the training loss in eval mode after each epoch", action="store_true")
parser.add_argument("--verbose", help="print classification metrics per each class, rather than aggregate metrics", action="store_true")
parser.add_argument("--method", default="nn", choices=["lr", "rf", "nn"], help="choice of learning method to use. lr and rf are baselines, so other command line argumetns do not affect them")
parser.add_argument("--output-latents", default=None, help="nn latent embeddings of all cells for visualization", dest="output_latents")
parser.add_argument("--loss-plot", default=None, help="outfile to store loss curves", dest="loss_plot")

def main(args):
    # Load in the data from the specified datafile
    if args.data.endswith('.npy'):
        cells = np.load(args.data, allow_pickle=True).item()
        adata = sc.AnnData(X=cells['UMI'].toarray(), obs={'celltype': cells['classes']}, var={'gene_id': cells['gene_ids']})
    elif args.data.endswith('h5ad'):
        adata = sc.read_h5ad(args.data)

    # Normalize the data, if it hasn't been normalized already.
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    torch.manual_seed(args.seed)

    # Setup the data and compute train / validation / test splits and scale features
    X, y = adata.X, adata.obs['celltype']
    print(f"Provided featurization has {X.shape} shape for X and {y.shape} shape for y")

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, stratify=y_train_val, random_state=args.seed)

    if args.method == 'nn':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        wandb.init(project=args.wandb_proj, group=args.group)

        # convert to tensors and make a dataloader. 
        # this dataset is small enough to completely load into memory for almost any gpu or cpu. 
        # if this dataset were much larger, we would instead move each batch to the gpu as needed.
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        y_train_tensor = torch.LongTensor(pd.Categorical(y_train).codes.copy()).to(device)
        y_val_tensor = torch.LongTensor(pd.Categorical(y_val).codes.copy()).to(device)
        y_test_tensor = torch.LongTensor(pd.Categorical(y_test).codes.copy()).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # Initialize the model
        input_size = X_train_tensor.shape[1]
        num_classes = len(np.unique(y))
        model = SCNet(input_size, num_classes, embed_size=args.embed_size, num_hidden_layers=args.num_hidden_layers, dropout=args.dropout, lr=args.lr)
        wandb.watch(model, log_freq=100)

        model = model.to(device)

        # Train the model given config
        num_epochs = args.epochs
        train_losses, val_losses = [], []
        pbar = tqdm(range(num_epochs), desc="Training", unit="epoch") 
        for epoch in pbar:
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_loss = model.train_step(batch_x, batch_y)
                train_loss += batch_loss / len(train_loader)

            if args.recompute_train_loss:
                train_loss = model.eval_step(X_train_tensor, y_train_tensor) # with a bigger dataset, we would do this in batches
            val_loss = model.eval_step(X_val_tensor, y_val_tensor)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            pbar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
        pbar.close()

        # Evaluate the neural model on all three sets: train, validation, and test
        label_encoder = pd.Categorical(y_train)
        for (X_tensor, y_tensor, set_name) in [
            (X_train_tensor, y_train_tensor, 'train'), 
            (X_val_tensor, y_val_tensor, 'val'),
            (X_test_tensor, y_test_tensor, 'test')
        ]:
            # Evaluate the model on the the respective set
            set_loss = model.eval_step(X_tensor, y_tensor)
            predictions = model.classify(X_tensor.to(device))
            pred_y = predictions.cpu().numpy()
            true_y = y_tensor.cpu().numpy()

            # Convert numeric labels back to original cell type names
            pred_y_names = label_encoder.categories[pred_y]
            true_y_names = label_encoder.categories[true_y]

            # Get probabilities for each class
            with torch.no_grad():
                nn_probs = torch.softmax(model(X_tensor), dim=1).cpu().numpy()

            # Binarize the labels for multi-class aupr and auroc. compute separately from other methods to track to wandb
            nn_y_true_bin = label_binarize(true_y_names, classes=label_encoder.categories.values)

            # Compute metrics for this set, and only save roc and pr curves if test set
            metrics = compute_metrics(true_y_names, pred_y_names, nn_y_true_bin, nn_probs, method='nn', save_curve=set_name == 'test')
            clf_report = metrics['classification_report']
            # delete the key from metrics to log to wandb / print
            del metrics['classification_report']

            wandb.log({
                "epoch": num_epochs, 
                f"{set_name}_loss": set_loss, 
                **metrics,
            })

            print(f"Results for the {set_name} set:")
            if args.verbose:
                print('\t', clf_report.to_csv(sep='\t'))

            print('\t', {
                f"{set_name}_loss": set_loss,
                **metrics
            })

        # Save the model's latent embeddings for visualization, if the user wants this.
        if args.output_latents is not None:
            # concatenate all the X tensors together to get the full dataset
            X_tensor = torch.cat([X_train_tensor, X_val_tensor, X_test_tensor], dim=0)
            cell_names = np.concatenate([y_train, y_val, y_test]) # this is the original cell type names
            model.save_latents(X_tensor, cell_names, args.output_latents)

        if args.loss_plot is not None:
            # Plot the model's output losses
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title("Training and Validation Losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(args.loss_plot)

    else:
        if args.method == 'lr':
            model = LogisticRegression(multi_class='ovr', max_iter=1000)
        else:
            assert args.method == 'rf'
            model = RandomForestClassifier(random_state=args.seed) 

        model.fit(X_train, y_train)

        # For the baseline methods we only computing metrics on the test set
        pred_y_names = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        true_y_names = y_test

        # Binarize the labels for AUROC and AUPR calculation
        y_test_bin = label_binarize(y_test, classes=model.classes_)

        # Compute metrics for this set
        metrics = compute_metrics(true_y_names, pred_y_names, y_test_bin, probabilities, method=f'{args.method}', save_curve=True)
        clf_report = metrics['classification_report']
        # delete the key from metrics to log to wandb / print
        del metrics['classification_report']

        print(f"Results for the test set:")
        if args.verbose:
            print('\t', clf_report.to_csv(sep='\t'))

        print(f"Results for the test set:")
        print('\t', metrics)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

