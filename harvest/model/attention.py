import torch
import torch.nn.functional as F


class AttentionLayers:
    """Attention layer implementations for graph neural networks."""
    
    @staticmethod
    def graph_attention_layer(A: torch.Tensor, H: torch.Tensor, v: list) -> torch.Tensor:
        """Cell-to-cell attention layer."""
        h1 = torch.mm(H, v[0])
        h1 = A * h1
        h2 = torch.mm(H, v[1])
        h2 = A * torch.transpose(h2, 1, 0)
        logits = torch.add(h1, h2)
        attention = F.sigmoid(logits)
        attention = attention * A
        attention[attention == 0] = float('-inf')
        attention = torch.softmax(attention, dim=1)
        return attention
    
    @staticmethod
    def graph_attention_layer_g(A: torch.Tensor, H: torch.Tensor, v: list) -> torch.Tensor:
        """Cell-to-gene attention layer."""
        h1 = torch.mm(H, v[0])
        h1 = A * torch.transpose(h1, 1, 0)
        h2 = torch.mm(H, v[1])
        h2 = A * torch.transpose(h2, 1, 0)
        logits = torch.add(h1, h2)
        attention = F.sigmoid(logits)
        attention = attention * A
        attention[attention == 0] = float('-inf')
        attention = torch.softmax(attention, dim=1)
        return attention
    
    @staticmethod
    def graph_attention_layer_gg(A: torch.Tensor, H: torch.Tensor, v: list) -> torch.Tensor:
        """Gene-to-gene attention layer."""
        h1 = torch.mm(H, v[0])
        h1 = A * h1
        h2 = torch.mm(H, v[1])
        h2 = A * torch.transpose(h2, 1, 0)
        logits = torch.add(h1, h2)
        attention = F.sigmoid(logits)
        attention = attention * A
        attention[attention == 0] = float('-inf')
        attention = torch.softmax(attention, dim=1)
        return attention 