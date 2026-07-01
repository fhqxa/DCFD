import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """
    Adapter模块用于参数高效微调
    在预训练模型中插入小型神经网络，保持主干网络冻结

    架构：
    input -> Linear(down) -> ReLU -> Linear(up) -> output
    """

    def __init__(self, input_dim, output_dim, adapter_dim=64, dropout=0.1):
        """
        Args:
            input_dim: int - 输入特征维度
            adapter_dim: int - Adapter中间层维度（瓶颈维度）
            dropout: float - dropout概率
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapter_dim = adapter_dim
        if output_dim is None:
            output_dim = input_dim

        # 下投影层：从input_dim降到adapter_dim
        self.down_proj = nn.Linear(input_dim, adapter_dim)

        # 上投影层：从adapter_dim恢复到input_dim
        self.up_proj = nn.Linear(adapter_dim, output_dim)

        # 激活函数
        self.act_fn = nn.ReLU(inplace=True)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.BatchNorm1d(adapter_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化Adapter权重"""
        # 使用低秩分解的初始化方式
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor [B, N, D] 或 [B, D] - 输入特征

        Returns:
            output: Tensor - Adapter输出
        """
        # 残差连接
        residual = x

        # 通过Adapter
        x = self.down_proj(x)
        x = self.norm(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)

        # 残差连接
        if self.input_dim == self.output_dim:
            output = x + residual
        else:
            output = x

        return output

class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x


class ParallelAdapter(nn.Module):
    """
    并行Adapter模块
    与主分支并行的适配器，用于多分支融合
    """

    def __init__(self, input_dim, adapter_dim=64, dropout=0.1, activation='relu'):
        """
        Args:
            input_dim: int - 输入特征维度
            adapter_dim: int - Adapter中间层维度
            dropout: float - dropout概率
            activation: str - 激活函数类型 ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim

        # 下投影
        self.down_proj = nn.Linear(input_dim, adapter_dim)

        # 上投影
        self.up_proj = nn.Linear(adapter_dim, input_dim)

        # 激活函数
        if activation == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'tanh':
            self.act_fn = nn.Tanh()
        else:
            self.act_fn = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # LayerNorm用于稳定训练
        self.norm = nn.LayerNorm(input_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor - 输入特征

        Returns:
            output: Tensor - Adapter输出（与输入相加）
        """
        # 先通过LayerNorm稳定输入
        x_norm = self.norm(x)

        # Adapter路径
        adapter_out = self.down_proj(x_norm)
        adapter_out = self.act_fn(adapter_out)
        adapter_out = self.dropout(adapter_out)
        adapter_out = self.up_proj(adapter_out)
        adapter_out = self.dropout(adapter_out)

        # 与原输入残差连接
        output = x + adapter_out

        return output


class SequentialAdapter(nn.Sequential):
    """
    串行Adapter模块
    多个Adapter模块串行连接，增强表达能力
    """

    def __init__(self, input_dim, adapter_dims, dropout=0.1):
        """
        Args:
            input_dim: int - 输入特征维度
            adapter_dims: list - 多个Adapter的中间层维度列表
            dropout: float - dropout概率
        """
        layers = []
        in_dim = input_dim

        for adapter_dim in adapter_dims:
            layers.append(Adapter(in_dim, adapter_dim, dropout))
            in_dim = input_dim  # 保持输入维度不变（残差连接）

        super().__init__(*layers)


class AdapterMlp(nn.Module):
    """
    Adapter风格的MLP模块
    用于特征变换和降维
    """

    def __init__(self, input_dim, hidden_dims, output_dim=None,
                 dropout=0.1, activation='relu', last_activation=True):
        """
        Args:
            input_dim: int - 输入维度
            hidden_dims: list - 隐藏层维度列表
            output_dim: int, optional - 输出维度，None则等于input_dim
            dropout: float - dropout概率
            activation: str - 激活函数类型
            last_activation: bool - 是否在输出层应用激活函数
        """
        super().__init__()

        dims = [input_dim] + hidden_dims
        if output_dim is not None:
            dims.append(output_dim)
        else:
            output_dim = input_dim

        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            # 中间层添加激活和dropout
            if i < len(dims) - 2 or last_activation:
                if activation == 'relu':
                    self.layers.append(nn.ReLU(inplace=True))
                elif activation == 'gelu':
                    self.layers.append(nn.GELU())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                else:
                    self.layers.append(nn.Identity())

                self.layers.append(nn.Dropout(dropout))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x


class SparseAdapter(nn.Module):
    """
    稀疏Adapter模块
    只对部分通道/特征应用Adapter，降低参数数量
    """

    def __init__(self, input_dim, adapter_dim=64, dropout=0.1, sparsity=0.5):
        """
        Args:
            input_dim: int - 输入特征维度
            adapter_dim: int - Adapter中间层维度
            dropout: float - dropout概率
            sparsity: float - 稀疏度（0-1，越大越稀疏）
        """
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.sparsity = sparsity

        # 掩码，决定哪些通道使用Adapter
        self.register_buffer(
            'adapter_mask',
            torch.rand(input_dim) > sparsity
        )

        # 只有被选中的通道才有adapter参数
        self.active_dim = self.adapter_mask.sum().item()

        if self.active_dim > 0:
            self.down_proj = nn.Linear(self.active_dim, adapter_dim)
            self.up_proj = nn.Linear(adapter_dim, self.active_dim)

        self.act_fn = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if self.active_dim > 0:
            self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def update_sparsity(self, sparsity):
        """更新稀疏度"""
        self.sparsity = sparsity
        self.adapter_mask = torch.rand(self.input_dim) > sparsity
        self.active_dim = self.adapter_mask.sum().item()

        if self.active_dim > 0:
            self.down_proj = nn.Linear(self.active_dim, self.adapter_dim)
            self.up_proj = nn.Linear(self.adapter_dim, self.active_dim)
            self._init_weights()

    def forward(self, x):
        """
        Args:
            x: Tensor [B, D] - 输入特征

        Returns:
            output: Tensor - 适配后的特征
        """
        if self.active_dim == 0:
            return x

        # 选择活跃通道
        active_features = x[:, self.adapter_mask]

        # 通过Adapter
        adapted = self.down_proj(active_features)
        adapted = self.act_fn(adapted)
        adapted = self.dropout(adapted)
        adapted = self.up_proj(adapted)
        adapted = self.dropout(adapted)

        # 合并回原特征
        output = x.clone()
        output[:, self.adapter_mask] += adapted

        return output


class MultiScaleAdapter(nn.Module):
    """
    多尺度Adapter模块
    在不同尺度上应用Adapter，增强多尺度特征融合能力
    """

    def __init__(self, input_dim, scales=[1, 2, 4], adapter_dim=64, dropout=0.1):
        """
        Args:
            input_dim: int - 输入特征维度
            scales: list - 缩放比例列表
            adapter_dim: int - Adapter中间层维度
            dropout: float - dropout概率
        """
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales

        # 为每个尺度创建一个Adapter
        self.adapters = nn.ModuleList([
            Adapter(input_dim, adapter_dim, dropout)
            for s in scales
        ])

        # 特征融合层 - 融合所有Adapter的输出
        # 注意：Adapter的输出维度与输入维度相同（input_dim），不是adapter_dim
        fusion_dim = len(scales) * input_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for adapter in self.adapters:
            adapter._init_weights()

        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor - 输入特征 [B, D]

        Returns:
            output: Tensor - 多尺度适配后的特征
        """
        adapted_features = []

        for scale, adapter in zip(self.scales, self.adapters):
            # 对整个输入特征应用Adapter，但通过不同的缩放参数影响权重
            # 这里使用相同的输入，但在Adapter内部通过缩放比例来影响处理
            adapted = adapter(x)
            adapted_features.append(adapted)

        # 特征拼接
        concatenated = torch.cat(adapted_features, dim=-1)

        # 融合
        output = self.fusion(concatenated)

        # 残差连接
        return x + output
