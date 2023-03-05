import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels):  # 构造的时候必须输入in，out
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):  # 调用的时候必须输入 x, edge_index
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        #### Steps 1-2 are typically computed before message passing takes place.
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix. 压缩 node feature
        x = self.lin(x)

        #### Steps 3-5 can be easily processed using the torch_geometric.nn.MessagePassing base class.
        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)  # 得到x_j

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)  # [N, ] dtype是数据类型
        deg_inv_sqrt = deg.pow(-0.5)  # [N(-0.5), ]
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # (N(-0.5),N(-0.5),E)

        # step 4：Aggregation.
        return norm.view(-1, 1) * x_j  # [N, E]*[E, out_channels]=[N, out_channels]

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_0 \cdot \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 bias=True,
                 **kwargs):
        super(RGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # size = self.num_bases * self.in_channels
        # uniform(size, self.basis)
        # uniform(size, self.att)
        # uniform(size, self.root)
        # xavier_normal_(self.basis, gain=calculate_gain('relu'))
        # xavier_normal_(self.root, gain=calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.basis)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.root)

    def forward(self, x, edge_index, edge_attr, edge_norm=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_attr, edge_norm):
        # x_j.shape = [26304, 2]
        # edge_index_j.shape = [26304]
        # edge_attr.shape = [26304, 3]

        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)

        out = torch.einsum('bi,rio->bro', x_j, w)
        out = (out * edge_attr.unsqueeze(2)).sum(dim=1)
        # out.shape = [25712, 768])
        # print(out.size(), edge_attr.unsqueeze(2).size())
        # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        # aggr_out.shape = [640, 768]
        # x.shape = [640, 2]
        if x is None:
            out = aggr_out + self.root
        else:
            out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
