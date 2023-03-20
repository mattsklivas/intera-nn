import torch
import torch.nn as nn

# PyTorch LSTM Model : https://www.youtube.com/watch?v=0_PgWWmauHk
# PyTorch Deployment Flask : https://www.youtube.com/watch?v=bA7-DEtYCNM

class AslNeuralNetwork(nn.Module):
    def __init__(self, input_size=201, lstm_hidden_size=200, fc_hidden_size=128, output_size=62, num_lstm_layers=4):
        # Call Neural network module initialization
        super(AslNeuralNetwork, self).__init__()

        # Define constants
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_hidden_size = fc_hidden_size
        
        # Create device with gpu support
        self.device = torch.device('cpu') # change accordingly

        # Define neural network architecture and activiation function
        # Long Short Term Memory (Lstm) and Fully Connected (Fc) Layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, self.num_lstm_layers, batch_first=True, bidirectional=True, dropout=0.25)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, output_size)

        # Scaled Exponential Linear Units (SELU) Activation
        # -> Self-normalization (internal normalization) by converging to mean and unit variance of zero
        self.relu = nn.SELU() #nn.LeakyReLU() #
        self.dropout = nn.AlphaDropout(0.25) #nn.Dropout(dropout_rate) #
        
    # Define forward pass, passing input x
    def forward(self, x):
        # Define initial tensors for hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_lstm_layers * 2, batch_size, self.lstm_hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_lstm_layers * 2, batch_size, self.lstm_hidden_size).to(self.device) 
        
        # Pass input with initial tensors to lstm layers
        out_lstm, _ = self.lstm(x, (h0, c0))
        out_relu = self.relu(out_lstm)
        
        # Many-One Architecture: Pass only last timestep to fc layers
        in_fc1 = out_relu[:, -1, :]
        in_fc1 = self.dropout(in_fc1)
        
        out_fc1 = self.fc1(in_fc1)
        in_fc2 = self.relu(out_fc1)
        in_fc2 = self.dropout(in_fc2)

        out = self.fc2(in_fc2)

        return out
