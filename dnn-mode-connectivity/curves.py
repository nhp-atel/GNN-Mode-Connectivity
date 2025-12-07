import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom


class Bezier(Module):
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)


class PolyChain(Module):
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class CurveModule(Module):

    def __init__(self, fix_points, parameter_names=()):
        super(CurveModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, '%s_%d' % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t


class Linear(CurveModule):

    def __init__(self, in_features, out_features, fix_points, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(torch.Tensor(out_features, in_features), requires_grad=not fixed)
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.linear(input, weight_t, bias_t)


class Conv2d(CurveModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(
                    torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                    requires_grad=not fixed
                )
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.conv2d(input, weight_t, bias_t, self.stride,
                        self.padding, self.dilation, self.groups)


class _BatchNorm(CurveModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'weight_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('weight_%d' % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i in range(self.num_bends):
                getattr(self, 'weight_%d' % i).data.uniform_()
                getattr(self, 'bias_%d' % i).data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight_t, bias_t,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class GATConv(CurveModule):
    """
    Graph Attention Network layer with Bezier curve parameter interpolation.

    Implements GAT attention mechanism where all parameters (linear transformations
    and attention weights) are interpolated along a Bezier curve in parameter space.
    """

    def __init__(self, in_channels, out_channels, fix_points, heads=1,
                 concat=True, negative_slope=0.2, dropout=0.0, bias=True):
        # Register parameter names: lin_weight, lin_bias, att_src, att_dst
        super(GATConv, self).__init__(fix_points, ('lin_weight', 'lin_bias', 'att_src', 'att_dst'))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Register linear transformation parameters for each bend
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'lin_weight_%d' % i,
                Parameter(
                    torch.Tensor(heads * out_channels, in_channels),
                    requires_grad=not fixed
                )
            )
            if bias:
                self.register_parameter(
                    'lin_bias_%d' % i,
                    Parameter(torch.Tensor(heads * out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter('lin_bias_%d' % i, None)

        # Register attention parameters for each bend
        # att_src and att_dst are used to compute attention scores
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'att_src_%d' % i,
                Parameter(torch.Tensor(1, heads, out_channels), requires_grad=not fixed)
            )
            self.register_parameter(
                'att_dst_%d' % i,
                Parameter(torch.Tensor(1, heads, out_channels), requires_grad=not fixed)
            )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot/Xavier initialization"""
        gain = torch.nn.init.calculate_gain('relu')
        for i in range(self.num_bends):
            # Initialize linear weights
            torch.nn.init.xavier_uniform_(getattr(self, 'lin_weight_%d' % i), gain=gain)
            lin_bias = getattr(self, 'lin_bias_%d' % i)
            if lin_bias is not None:
                torch.nn.init.zeros_(lin_bias)

            # Initialize attention parameters
            torch.nn.init.xavier_uniform_(getattr(self, 'att_src_%d' % i), gain=gain)
            torch.nn.init.xavier_uniform_(getattr(self, 'att_dst_%d' % i), gain=gain)

    def forward(self, x, edge_index, coeffs_t):
        """
        Forward pass with parameter interpolation.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            coeffs_t: Bezier coefficients at curve point t

        Returns:
            Node features [num_nodes, heads * out_channels] if concat=True
            Node features [num_nodes, out_channels] if concat=False
        """
        # Interpolate parameters at curve point t
        lin_weight_t, lin_bias_t, att_src_t, att_dst_t = self.compute_weights_t(coeffs_t)

        H, C = self.heads, self.out_channels

        # Linear transformation: x -> Wx
        x_transformed = F.linear(x, lin_weight_t, lin_bias_t)  # [num_nodes, heads * out_channels]
        x_transformed = x_transformed.view(-1, H, C)  # [num_nodes, heads, out_channels]

        # Compute attention scores
        # alpha_src: [num_nodes, heads, 1]
        # alpha_dst: [num_nodes, heads, 1]
        alpha_src = (x_transformed * att_src_t).sum(dim=-1, keepdim=True)
        alpha_dst = (x_transformed * att_dst_t).sum(dim=-1, keepdim=True)

        # Get edge indices
        row, col = edge_index[0], edge_index[1]  # row: source nodes, col: target nodes

        # Compute attention coefficients: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # For each edge (i -> j): alpha_i + alpha_j
        alpha = alpha_src[row] + alpha_dst[col]  # [num_edges, heads, 1]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.squeeze(-1)  # [num_edges, heads]

        # Softmax normalization per target node
        # For each target node j, normalize over all incoming edges
        alpha = self._softmax(alpha, col, x.size(0))  # [num_edges, heads]

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message passing: aggregate neighbor features weighted by attention
        # out_j = sum_{i in N(j)} alpha_ij * Wh_i
        out = torch.zeros(x.size(0), H, C, dtype=x.dtype, device=x.device)
        x_transformed_src = x_transformed[row]  # [num_edges, heads, out_channels]

        # Weight by attention and aggregate
        for i in range(x.size(0)):
            # Find all edges pointing to node i
            mask = col == i
            if mask.any():
                # Get source node features and attention weights for edges pointing to i
                src_features = x_transformed_src[mask]  # [num_incoming_edges, heads, out_channels]
                attn_weights = alpha[mask]  # [num_incoming_edges, heads]
                # Aggregate: sum over incoming edges
                out[i] = (src_features * attn_weights.unsqueeze(-1)).sum(dim=0)

        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, H * C)  # [num_nodes, heads * out_channels]
        else:
            out = out.mean(dim=1)  # [num_nodes, out_channels]

        return out

    def _softmax(self, src, index, num_nodes):
        """
        Softmax normalization per target node.

        For each target node, normalize attention scores over incoming edges.
        """
        # Compute max for numerical stability
        out = src - self._scatter_max(src, index, dim=0, dim_size=num_nodes)[index]
        out = out.exp()
        out = out / (self._scatter_sum(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
        return out

    def _scatter_max(self, src, index, dim, dim_size):
        """Helper: scatter max operation"""
        out = torch.full((dim_size, src.size(1)), float('-inf'), dtype=src.dtype, device=src.device)
        out = out.scatter_reduce(dim, index.unsqueeze(-1).expand_as(src), src, reduce='amax', include_self=False)
        return out

    def _scatter_sum(self, src, index, dim, dim_size):
        """Helper: scatter sum operation"""
        out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
        out = out.scatter_add(dim, index.unsqueeze(-1).expand_as(src), src)
        return out


class CurveNet(Module):
    def __init__(self, num_classes, curve, architecture, num_bends, fix_start=True, fix_end=True,
                 architecture_kwargs={}):
        super(CurveNet, self).__init__()
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]
        
        self.curve = curve
        self.architecture = architecture

        self.l2 = 0.0
        self.coeff_layer = self.curve(self.num_bends)
        self.net = self.architecture(num_classes, fix_points=self.fix_points, **architecture_kwargs)
        self.curve_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)

    def import_base_parameters(self, base_model, index):
        """
        Import parameters from a base model into a specific bend of the curve model.
        Uses name-based matching for robustness.
        """
        base_dict = dict(base_model.named_parameters())
        curve_dict = dict(self.net.named_parameters())

        for base_name, base_param in base_dict.items():
            # Map base parameter name to curve parameter name with index suffix
            curve_name = f"{base_name}_{index}"

            if curve_name in curve_dict:
                # Direct match: base.weight -> base.weight_0
                curve_dict[curve_name].data.copy_(base_param.data)
            else:
                # For debugging: print unmatched parameters
                # print(f"Warning: Could not find curve parameter {curve_name} for base parameter {base_name}")
                pass

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.net._all_buffers(), base_model._all_buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        """
        Initialize intermediate curve points by linear interpolation between endpoints.
        Uses name-based grouping to handle parameters of different shapes correctly.
        """
        # Group parameters by base name (without _0, _1, _2 suffix)
        param_dict = dict(self.net.named_parameters())
        param_groups = {}

        for name, param in param_dict.items():
            # Extract base name by removing _0, _1, _2, etc. suffix
            if '_' in name:
                parts = name.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_name = parts[0]
                    index = int(parts[1])

                    if base_name not in param_groups:
                        param_groups[base_name] = {}
                    param_groups[base_name][index] = param

        # Interpolate within each parameter group
        for base_name, indexed_params in param_groups.items():
            # Only interpolate if we have all num_bends parameters
            if len(indexed_params) == self.num_bends:
                for j in range(1, self.num_bends - 1):
                    alpha = j * 1.0 / (self.num_bends - 1)
                    start_param = indexed_params[0]
                    end_param = indexed_params[self.num_bends - 1]
                    mid_param = indexed_params[j]

                    # Linear interpolation: param_j = alpha * end + (1 - alpha) * start
                    mid_param.data.copy_(
                        alpha * end_param.data + (1.0 - alpha) * start_param.data
                    )

    def weights(self, t):
        coeffs_t = self.coeff_layer(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def forward(self, *args, t=None, **kwargs):
        """
        Forward pass through curve model.

        Supports both image and graph inputs:
        - Images: forward(input, t=None)
        - Graphs: forward(x, edge_index, batch, t=None)
        """
        # Determine first tensor for device and uniform sampling
        first_arg = args[0] if len(args) > 0 else next(iter(kwargs.values()))

        if t is None:
            t = first_arg.data.new(1).uniform_()
        coeffs_t = self.coeff_layer(t)
        output = self.net(*args, coeffs_t, **kwargs)
        self._compute_l2()
        return output


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2
