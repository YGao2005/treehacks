import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FCNetwork, self).__init__()
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            #layers.append(nn.Dropout1d(p=0.2))
            layers.append(nn.ReLU())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_and_preprocess_data(data_path, val_ratio=0.2, random_state=42):

    # NOTE: Load 3 datasets separately
    # add their labels
    # add and mix them together

    all_instances = pd.DataFrame()
    ppg_filenames = ['regular', 'irregular', 'afib']
    label = 0
    for ppg_filename in ppg_filenames:
        in_file = ppg_filename + "_ppg.csv"
        df = pd.read_csv(in_file, index_col=None)
        df['label'] = 0 if ppg_filename == 'regular' else 1
        all_instances = pd.concat([all_instances, df], ignore_index = True)
        label += 1
    if not os.path.exists('dataset_ppg.csv'):
        all_instances.to_csv('dataset_ppg.csv', index=False)

    df = all_instances
    # NOTE: Separate features and labels
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # NOTE: Split into train and validation sets
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=val_ratio, random_state=random_state, stratify=y)

    # NOTE: Split into train and validation sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.2, random_state=random_state, stratify=y_val_test)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def create_dataloaders(X_train, X_val, y_train, y_val, batch_size):
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, output_dir):
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f'Checkpoint saved at epoch {epoch+1}')

def main():
    parser = argparse.ArgumentParser(description='Train a fully connected network for multiclass classification')
    parser.add_argument('--data_path', type=str, default='./heartbeat_data/', help='Path to input CSV file')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 32], help='List of hidden layer sizes')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--optimizer', default='sgd', help='optimizer of network')
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(
        args.data_path, args.val_ratio, args.seed)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, X_val, y_train, y_val, args.batch_size)
    
    # Initialize model
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FCNetwork(input_size, args.hidden_sizes, output_size).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                args.epochs, device, args.output_dir)


if __name__ == '__main__':
    main()
