import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd


class VAE(nn.Module):
    
    def __init__(self, input_dim, intermediate_dim, latent_dim, learning_rate, name='model', **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        self.name = name
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.threshold = None
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc_log_var = nn.Linear(intermediate_dim, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, input_dim)
    
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var
    
    
    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std * 0.25
    
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))
    
    
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z)
        return x_decoded, z_mean, z_log_var
    
    
    def loss_function(self, x, x_decoded, z_mean, z_log_var):
        reconstruction_loss = F.mse_loss(x_decoded, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return reconstruction_loss + kl_loss
    
    
    def train_model(self, x_train: pd.DataFrame, epochs, batch_size, save_dir=None, verbose=False):
        dataset = torch.utils.data.TensorDataset(torch.tensor(x_train.to_numpy(), dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                self.optimizer.zero_grad()
                x_decoded, z_mean, z_log_var = self(x)
                loss = self.loss_function(x, x_decoded, z_mean, z_log_var)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if verbose:
                print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader.dataset)}")
                
        model_path = os.path.join(save_dir, self.name + ".pt")
        torch.save(self.state_dict(), model_path)
    
        self.determine_classification_threshold(x_train.to_numpy())
    
    
    def calculate_reconstruction_error(self, data):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(data, dtype=torch.float32)
            x_decoded, _, _ = self(x)
            return torch.mean(torch.abs(x - x_decoded), dim=1).numpy()
    
    
    def determine_classification_threshold(self, x_train):
        mae_train = self.calculate_reconstruction_error(x_train)
        self.threshold_max = np.max(mae_train)
        self.threshold = np.percentile(mae_train, 99)
        self.threshold_90 = np.percentile(mae_train, 90)
        
        
    def get_recon_error(self, data):
        self.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32)
            recon_data = self(data)[0]
            return torch.abs(data - recon_data).numpy()
    
    
    def predict_anomaly(self, data):
        mae_data = self.calculate_reconstruction_error(data)
        pred = (mae_data > self.threshold).astype(int)
        return pred, mae_data
    
    
    def predict_anomaly_90(self, data):
        mae_data = self.calculate_reconstruction_error(data)
        pred = (mae_data > self.threshold_90).astype(int)
        return pred, mae_data
