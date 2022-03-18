import torch
from torch import nn


class L1SparsityLoss(nn.Module):
    """
    L1 Norm sparsity loss for node and edge weights over echo-graph

    Attributes
    ----------
    encoder_type: str, indicates the type of attention uses. Can be one of node, edge, node&edge or none.
    device: torch.device, device to use

    Methods
    -------
    forward(node_x, edge_x): computes the sparsity loss
    """
    def __init__(self,
                 encoder_type: str,
                 device: torch.device = torch.device('cpu')):
        """
        :param encoder_type: str, indicates the type of attention uses. Can be one of node, edge, node&edge or none.
        :param device: torch.device, device to use
        """

        super().__init__()

        assert encoder_type in ['node', 'edge', 'node&edge', 'none'], 'Loss type must be one of [node, edge, node&edge]'

        self.encoder_type = encoder_type
        self.device = device

    def forward(self, node_x: torch.tensor = None, edge_x: torch.tensor = None):
        """
        Computes the sparsity loss

        :param node_x: torch.tensor, weights over nodes of shape N*num_nodes*1
        :param edge_x: torch.tensor, weights over edges of shape N*num_edges*1
        :return: L1 norm of nodes and edges
        """

        if self.encoder_type == 'node' or self.encoder_type == 'node&edge':
            node_loss = torch.norm(node_x, p=1, dim=1).mean()
        else:
            node_loss = torch.tensor(0, device=self.device)

        if self.encoder_type == 'edge' or self.encoder_type == 'node&edge':
            edge_loss = torch.norm(edge_x, p=1, dim=1).mean()
        else:
            edge_loss = torch.tensor(0, device=self.device)

        return node_loss, edge_loss
