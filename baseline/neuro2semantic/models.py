import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=1536, hidden_dim=1024, num_layers=2, dropout_rate=0.3):
        super(NeuralEncoder, self).__init__()
        
        self.rnn = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, masks):
        """
        Forward pass to generate embeddings from input sequences.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_dim].
            masks (torch.Tensor): Mask tensor of shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: Normalized embeddings of shape [batch_size, embedding_dim].
        """
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out * masks.unsqueeze(-1)

        rnn_out_sum = torch.sum(rnn_out, dim=1)
        mask_sum = torch.sum(masks, dim=1, keepdim=True)
        rnn_out_mean = rnn_out_sum / mask_sum

        embedding = self.fc(rnn_out_mean)
        embedding = self.dropout(embedding)
        
        if embedding.size(0) > 1:
            embedding = self.bn(embedding)
        
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

