import argparse
import torch
import wandb

import numpy as np
import pandas as pd
import scanpy as sc

from model import SCNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description="Cell Type Identification Training.")
parser.add_argument("--wandb-proj", default="Interview", help="Weights & Biases Project", dest="wandb_proj")
parser.add_argument("--data", default="cells.npy", help="scRNA data to use for training", dest="data")
parser.add_argument("--seed", type=int, default=0, help="Random seed to use for train / val / test splits", dest="seed")
parser.add_argument("--lr", "--learning-rate", type=float, default=3e-4, help="initial learning rate", dest="lr",)
parser.add_argument("--embed-size", type=int, default=128, help="hidden dimension embedding size", dest="embed_size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train for", dest="epochs")
parser.add_argument("--num_hidden_layers", type=int, default=3, help="number of hidden layers to use in the model", dest="num_hidden_layers")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate", dest="dropout")
parser.add_argument("--output-latents", default="nn_latents.pt", help="nn latent embeddings of all cells for visualization", dest="output_latents")
parser.add_argument("--method", default="nn", choices=["lr", "rf", "nn"], help="choice of learning method to use. lr and rf are baselines, so other command line argumetns do not affect them")

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
        wandb.init(project=args.wandb_proj)

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
        model = SCNet(input_size, num_classes, embed_size=args.embed_size, num_hidden_layers=args.num_hidden_layers, dropout=args.dropout)
        wandb.watch(model, log_freq=100)

        model = model.to(device)

        # Train the model given config
        num_epochs = args.epochs
        for epoch in range(num_epochs):
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_loss = model.train_step(batch_x, batch_y)
                train_loss += batch_loss / len(train_loader)

            val_loss = model.eval_step(X_val_tensor, y_val_tensor)
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        
        # Evaluate the model on the heldout test dataset
        test_loss = model.eval_step(X_test_tensor, y_test_tensor)
        predictions = model.classify(X_test_tensor.to(device))
        pred_y = predictions.cpu().numpy()
        test_y = y_test_tensor.cpu().numpy()

        # Convert numeric labels back to original cell type names
        label_encoder = pd.Categorical(y_train)
        pred_y_names = label_encoder.categories[pred_y]
        test_y_names = label_encoder.categories[test_y]

        report = classification_report(test_y_names, pred_y_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        # Get probabilities for each class
        with torch.no_grad():
            nn_probs = torch.softmax(model(X_test_tensor), dim=1).cpu().numpy()

        # Binarize the labels for multi-class aupr and auroc
        nn_y_test_bin = label_binarize(test_y_names, classes=pd.Categorical(y_train).categories.values)

        # Compute micro-averaged AUROC
        auroc = roc_auc_score(nn_y_test_bin, nn_probs, average='micro', )

        # Compute micro-averaged AUPR
        micro_aupr = average_precision_score(nn_y_test_bin, nn_probs, average='micro')
        macro_aupr = average_precision_score(nn_y_test_bin, nn_probs, average='macro')

        wandb.log({"epoch": num_epochs, "test_loss": test_loss, "auroc": auroc, "micro_aupr": micro_aupr, "macro_aupr": macro_aupr})

    else:
        if args.method == 'lr':
            model = LogisticRegression(multi_class='ovr', max_iter=1000)
        else:
            assert args.method == 'rf'
            model = RandomForestClassifier(random_state=args.seed) 

        model.fit(X_train, y_train)
        pred_y_names = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        test_y_names = y_test

        # Generate classification report
        report = classification_report(test_y_names, pred_y_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        # Binarize the labels for AUROC and AUPR calculation
        classes = model.classes_
        y_test_bin = label_binarize(y_test, classes=classes)

        # Compute micro-averaged AUROC and AUPR
        auroc = roc_auc_score(y_test_bin, probabilities, average='micro')
        micro_aupr = average_precision_score(y_test_bin, probabilities, average='micro')
        macro_aupr = average_precision_score(y_test_bin, probabilities, average='macro')


    print(df_report.to_csv(sep='\t'))
    print(f"Micro-averaged AUROC: {auroc:.4f}")
    print(f"Micro-averaged AUPR: {micro_aupr:.4f}")
    print(f"Macro-averaged AUPR: {macro_aupr:.4f}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
