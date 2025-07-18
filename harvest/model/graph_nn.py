import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .attention import AttentionLayers


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network with attention mechanisms for spatial transcriptomics."""
    
    def __init__(self, hidden_dims_c: list, hidden_dims_g: list, 
                 adjacency_matrix: torch.Tensor, mi_score: torch.Tensor, 
                 cg_expr: torch.Tensor, coor_ratio: float = 1.0):
        """
        Initialize the Graph Neural Network.
        
        Parameters:
        -----------
        hidden_dims_c : list
            Dimensions of cell representation layers
        hidden_dims_g : list
            Dimensions of gene representation layers
        adjacency_matrix : torch.Tensor
            Cell-to-cell adjacency matrix
        mi_score : torch.Tensor
            Gene-to-gene mutual information matrix
        cg_expr : torch.Tensor
            Cell-to-gene expression matrix
        coor_ratio : float
            Coordination ratio parameter
        """
        super(GraphNeuralNetwork, self).__init__()
        self.n_layers = len(hidden_dims_c) - 1
        self.adj = adjacency_matrix
        self.mi = mi_score
        self.expr = cg_expr
        self.coor_ratio = coor_ratio
        
        # Attention dictionaries
        self.atten = {}
        self.atten_g = {}
        self.atten_gg = {}
        
        # Weight matrices
        self.W_c = nn.ParameterList([
            Parameter(torch.FloatTensor(hidden_dims_c[i], hidden_dims_c[i+1]))
            for i in range(self.n_layers)
        ])
        self.W_g = nn.ParameterList([
            Parameter(torch.FloatTensor(hidden_dims_g[i], hidden_dims_g[i+1]))
            for i in range(self.n_layers)
        ])
        self.W_gc = nn.ParameterList([
            Parameter(torch.FloatTensor(hidden_dims_g[1], hidden_dims_g[2]))
        ])
        
        # Attention parameters
        self.Ws_att = nn.ParameterList([])
        self.Ws_att_g = nn.ParameterList([])
        self.Ws_att_gg = nn.ParameterList([])
        
        for i in range(self.n_layers-1):
            v = nn.ParameterList([])
            v.append(Parameter(torch.FloatTensor(hidden_dims_c[i+1], 1)))
            v.append(Parameter(torch.FloatTensor(hidden_dims_c[i+1], 1)))
            self.Ws_att.append(v)
        
        for i in range(self.n_layers-1):
            v = nn.ParameterList([])
            v.append(Parameter(torch.FloatTensor(hidden_dims_g[i+1], 1)))
            v.append(Parameter(torch.FloatTensor(hidden_dims_g[i+1], 1)))
            self.Ws_att_g.append(v)
        
        for i in range(self.n_layers-1):
            v = nn.ParameterList([])
            v.append(Parameter(torch.FloatTensor(hidden_dims_c[i+1], 1)))
            v.append(Parameter(torch.FloatTensor(hidden_dims_c[i+1], 1)))
            self.Ws_att_gg.append(v)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize network parameters with Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.Ws_att[0][0])
        torch.nn.init.xavier_uniform_(self.Ws_att[0][1])
        torch.nn.init.xavier_uniform_(self.Ws_att_g[0][0])
        torch.nn.init.xavier_uniform_(self.Ws_att_g[0][1])
        torch.nn.init.xavier_uniform_(self.Ws_att_gg[0][0])
        torch.nn.init.xavier_uniform_(self.Ws_att_gg[0][1])
        torch.nn.init.xavier_uniform_(self.W_c[0])
        torch.nn.init.xavier_uniform_(self.W_c[1])
        torch.nn.init.xavier_uniform_(self.W_gc[0])
        torch.nn.init.xavier_uniform_(self.W_g[0])
        torch.nn.init.xavier_uniform_(self.W_g[1])
    
    def forward(self, X_c: torch.Tensor, X_g: torch.Tensor):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X_c : torch.Tensor
            Cell features
        X_g : torch.Tensor
            Gene features
        
        Returns:
        --------
        tuple : (hidden, hidden_coor, reX_c, reX_g)
            Final embeddings and reconstructed features
        """
        H_g = X_g
        H_c = X_c
        for layer in range(self.n_layers):
            H_c, H_g, H_c_coor, H_c_expr = self.encoder(H_c, H_g, self.mi, self.adj, self.expr, layer)
            if layer != self.n_layers-1:
                H_c = F.elu(H_c)
                H_g = F.elu(H_g)
                H_c_coor = F.elu(H_c_coor)
                H_c_expr = F.elu(H_c_expr)
        
        hidden = H_c
        hidden_coor = H_c_coor
        
        for layer in range(self.n_layers-1, -1, -1):
            H_c_coor, H_c_expr = self.decoder(H_c_coor, H_c_expr, self.adj, self.expr, self.mi, layer)
            if layer != 0:
                H_c_coor = F.elu(H_c_coor)
                H_c_expr = F.elu(H_c_expr)
        
        reX_c = H_c_coor
        reX_g = H_c_expr
        
        return hidden, hidden_coor, reX_c, reX_g
    
    def encoder(self, X_c: torch.Tensor, X_g: torch.Tensor, mi: torch.Tensor, 
               adj: torch.Tensor, expr: torch.Tensor, layer: int):
        """Encoder part of the network."""
        if layer != self.n_layers-1:
            z_g = torch.mm(X_g, self.W_g[layer])
            self.atten_gg[layer] = AttentionLayers.graph_attention_layer_gg(mi, z_g, self.Ws_att_gg[0])
            z_g = torch.mm(self.atten_gg[layer], z_g)
            z_c = torch.mm(X_c, self.W_c[layer])
            z_c_coor = torch.tensor(0, dtype=torch.float)
            z_c_expr = torch.tensor(0, dtype=torch.float)
            return z_c, z_g, z_c_coor, z_c_expr
        
        elif layer == self.n_layers-1:
            self.atten[layer] = AttentionLayers.graph_attention_layer(adj, X_c, self.Ws_att[0])
            z_c_coor = torch.mm(self.atten[layer], X_c)
            z_c_coor = torch.mm(z_c_coor, self.W_c[layer])
            self.atten_g[layer] = AttentionLayers.graph_attention_layer_g(expr, X_g, self.Ws_att_g[0])
            z_c_expr = torch.mm(self.atten_g[layer], X_g)
            z_c_expr = torch.mm(z_c_expr, self.W_gc[0])
            z_c = self.coor_ratio * z_c_coor + (1 - self.coor_ratio) * z_c_expr
            return z_c, X_g, z_c_coor, z_c_expr
    
    def decoder(self, z_c_coor: torch.Tensor, z_c_expr: torch.Tensor, 
               adj: torch.Tensor, expr: torch.Tensor, mi: torch.Tensor, layer: int):
        """Decoder part of the network."""
        if layer == 0:
            D_c_coor = torch.mm(z_c_coor, torch.transpose(self.W_c[layer], 1, 0))
            D_expr = torch.mm(self.atten_gg[layer], z_c_expr)
            D_expr = torch.mm(D_expr, torch.transpose(self.W_g[0], 1, 0))
            return D_c_coor, D_expr
        else:
            D_c_coor = torch.mm(z_c_coor, torch.transpose(self.W_c[layer], 1, 0))
            D_c_coor = torch.mm(self.atten[layer], D_c_coor)
            D_expr = torch.mm(z_c_expr, torch.transpose(self.W_gc[0], 1, 0))
            D_expr = torch.mm(torch.transpose(self.atten_g[layer], 1, 0), D_expr)
            return D_c_coor, D_expr 