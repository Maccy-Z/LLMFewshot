from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn import GATConv

import torch.nn.functional as tf


class GATConvFunc(MessagePassing):
    def __init__(
            self,
            #in_channels: Union[int, Tuple[int, int]],
            #out_channels: int,
            #heads: int = 1,
            # concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            # add_self_loops: bool = True,
            # edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            # bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        #self.in_channels = in_channels
        #self.out_channels = out_channels
        #self.heads = heads
        # self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # self.add_self_loops = add_self_loops
        # self.edge_dim = edge_dim
        self.fill_value = fill_value

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                lin_weight, att_src, att_dst, bias):

        #H, C = self.heads, self.out_channels
        H, C = att_src.shape[-2], att_dst.shape[-1]

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        self.lin_src = tf.linear(x, lin_weight)
        xs = self.lin_src.view(-1, H, C)

        x = (xs, xs)
        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        # Identical to dot product here,

        alpha_src = (xs * att_src).sum(-1)
        alpha_dst = (xs * att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        # We only want to add self-loops for nodes that appear both as
        # source and target nodes:
        num_nodes = xs.size(0)

        # Add self-loops onto graph.
        edge_index, _ = remove_self_loops(
            edge_index, None)
        edge_index, _ = add_self_loops(
            edge_index, None, fill_value=self.fill_value,
            num_nodes=num_nodes)

        # Compute attention weights alpha.
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=None)
        self.save_alpha = alpha
        self.edge_index = edge_index
        # print(alpha)
        #
        # print(edge_index)
        #
        # exit(3)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        # Concatenate output
        out = out.view(-1, H * C)

        out = out + bias

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :class:`torch.Tensor`,
                a :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is a :obj:`torch.Tensor`, its :obj:`dtype`
                should be :obj:`torch.long` and its shape needs to be defined
                as :obj:`[2, num_messages]` where messages from nodes in
                :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is a :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :meth:`propagate`.
            size ((int, int), optional): The size :obj:`(N, M)` of the
                assignment matrix in case :obj:`edge_index` is a
                :class:`torch.Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size,
                                  kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)

        return out

    def edge_updater(self, edge_index: Adj, **kwargs):
        r"""The initial call to compute or update features for each edge in the
        graph.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :obj:`torch.Tensor`, a
                :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying graph
                connectivity/message passing flow.
                See :meth:`propagate` for more information.
            **kwargs: Any additional data which is needed to compute or update
                features for each edge in the graph.
        """
        size = self._check_input(edge_index, size=None)

        coll_dict = self._collect(self._edge_user_args, edge_index, size,
                                  kwargs)

        edge_kwargs = self.inspector.distribute('edge_update', coll_dict)
        out = self.edge_update(**edge_kwargs)

        return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


def main():
    # Compare the outputs of GATconv vs functional implementation
    in_dim, out_dim = 1, 2
    heads = 5

    linear_weight = torch.rand((out_dim * heads, in_dim), requires_grad=True)
    src = torch.rand((1, heads, out_dim))
    dst = torch.rand((1, heads, out_dim))
    bias = torch.rand(out_dim * heads)

    model = GATConvFunc()

    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                               [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    data = torch.tensor([[1], [30.], [1]], dtype=torch.float)

    func_out = model(data, edge_index, lin_weight=linear_weight, att_src=src, att_dst=dst, bias=bias)

    # Make a identical GATConv layer
    model2 = GATConv(in_dim, out_dim, heads=heads)

    model2.att_src = torch.nn.Parameter(src)
    model2.att_dst = torch.nn.Parameter(dst)
    model2.bias = torch.nn.Parameter(bias)
    model2.lin_src.weight = torch.nn.Parameter(linear_weight)
    base_out = model2(data, edge_index)
    print("Functional GATConv output")
    print(func_out)
    print()
    print("Normal GATConv output")
    print(base_out)
    # loss = torch.mean(base_out)
    # loss.backward()
    # print(model2.lin_src.weight.grad)

if __name__ == "__main__":
   # torch.manual_seed(123)
    main()

    # tensor([[0.4632, 0.5067, 0.4027],
    #         [0.1512, 0.1654, 0.1315],
    #         [-0.3556, -0.3889, -0.3091]], grad_fn= < AddBackward0 >)
