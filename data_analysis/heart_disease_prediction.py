import os
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

def load_and_preprocess_data(data_path, output_dir, val_ratio=0.2, random_state=42):

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

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
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
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.pth'))
            print(f'Checkpoint saved at epoch {epoch+1}')

def load_model(pt_path, input_size=3, hidden_sizes=[64, 32], output_size=2):
    model = FCNetwork(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    checkpoint = torch.load(pt_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def predict(instance, pt_dir, pt_file = 'checkpoint.pth', model=None):
    if not model:
        model = load_model(os.path.join(pt_dir, pt_file))
    scaler = joblib.load(os.path.join(pt_dir, 'scaler.pkl'))
    instance = torch.FloatTensor(instance).reshape(1, -1)
    instance = torch.FloatTensor(scaler.transform(instance))
    output = model(instance)
    res = torch.argmax(output)
    probs = F.softmax(output, dim=1).tolist()[0]
    label_names = ['regular', 'irregular']
    return label_names[res], probs

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
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    example = [72.66417352281226,48.668455142145376,46.93269563232047]
    #example = [82.19289340101524,235.42250031313927,173.97601912929474]
    if args.predict:
        res, probs = predict(example, args.output_dir)
        print("class:", res)
        print("prob:", probs)
        return

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(
        args.data_path, args.output_dir, args.val_ratio, args.seed)

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
