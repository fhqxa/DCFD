import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    基础多层感知机
    """

    def __init__(self, input_dim, hidden_dims, output_dim=None,
                 dropout=0.0, activation='relu', batch_norm=False):
        """
        Args:
            input_dim: int - 输入维度
            hidden_dims: list - 隐藏层维度列表
            output_dim: int, optional - 输出维度，None则等于input_dim
            dropout: float - dropout概率
            activation: str - 激活函数类型 ('relu', 'gelu', 'tanh', 'sigmoid', 'identity')
            batch_norm: bool - 是否使用BatchNorm
        """
        super().__init__()

        dims = [input_dim] + hidden_dims
        if output_dim is not None:
            dims.append(output_dim)
        else:
            output_dim = input_dim

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None

        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(linear)

            # BatchNorm
            if self.bns is not None:
                bn = nn.BatchNorm1d(dims[i + 1])
                self.bns.append(bn)

            # 激活函数
            if activation == 'relu':
                act = nn.ReLU(inplace=True)
            elif activation == 'gelu':
                act = nn.GELU()
            elif activation == 'tanh':
                act = nn.Tanh()
            elif activation == 'sigmoid':
                act = nn.Sigmoid()
            else:  # identity
                act = nn.Identity()

            self.layers.append(act)

            # Dropout
            if dropout > 0:
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
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x


class FastMLP(nn.Module):
    """
    快速MLP模块
    优化内存使用和计算速度
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2,
                 dropout=0.0, activation='gelu'):
        """
        Args:
            input_dim: int - 输入维度
            hidden_dim: int - 隐藏层维度
            output_dim: int - 输出维度
            num_layers: int - 层数
            dropout: float - dropout概率
            activation: str - 激活函数类型
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 创建MLP层
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        # 激活函数
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """前向传播"""
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # 除了最后一层，都应用激活和dropout
            if i < self.num_layers - 1:
                x = self.act(x)
                if self.dropout_layer is not None:
                    x = self.dropout_layer(x)

        return x


class ConditionalMLP(nn.Module):
    """
    条件MLP模块
    根据条件向量动态调整隐藏层
    """

    def __init__(self, input_dim, condition_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.0):
        """
        Args:
            input_dim: int - 输入维度
            condition_dim: int - 条件向量维度
            hidden_dim: int - 隐藏层维度
            output_dim: int - 输出维度
            num_layers: int - 层数
            dropout: float - dropout概率
        """
        super().__init__()
        self.num_layers = num_layers

        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * num_layers)
        )

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # MLP层（权重将由条件决定）
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, condition):
        """
        Args:
            x: Tensor - 输入特征
            condition: Tensor - 条件向量

        Returns:
            output: Tensor - 条件输出
        """
        batch_size = x.shape[0]

        # 编码条件
        condition_emb = self.condition_encoder(condition)
        # 重新reshape为每层的偏置
        condition_emb = condition_emb.view(batch_size, self.num_layers, -1)

        # 输入投影
        x = self.input_proj(x)

        # 逐层计算
        for i, layer in enumerate(self.hidden_layers):
            # 条件偏置
            bias = condition_emb[:, i, :]

            # 计算
            x = layer(x) + bias
            x = self.act(x)
            x = self.dropout(x)

        # 输出
        output = self.output_layer(x)

        return output


class LightweightMLP(nn.Module):
    """
    轻量级MLP模块
    用于移动设备和边缘计算
    """

    def __init__(self, input_dim, output_dim, hidden_ratio=4, dropout=0.1):
        """
        Args:
            input_dim: int - 输入维度
            output_dim: int - 输出维度
            hidden_ratio: float - 隐藏层维度比例 (hidden = max(1, input_dim // hidden_ratio))
            dropout: float - dropout概率
        """
        super().__init__()

        hidden_dim = max(1, input_dim // hidden_ratio)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播"""
        return self.layers(x)


class ResidualMLP(nn.Module):
    """
    带残差连接的MLP
    提高梯度流和训练稳定性
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2,
                 dropout=0.1, activation='relu'):
        """
        Args:
            input_dim: int - 输入维度
            hidden_dim: int - 隐藏层维度
            output_dim: int - 输出维度
            num_blocks: int - 残差块数量
            dropout: float - dropout概率
            activation: str - 激活函数类型
        """
        super().__init__()

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, activation)
            for _ in range(num_blocks)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 激活函数
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        """前向传播"""
        # 输入投影
        x = self.input_proj(x)
        x = self.act(x)

        # 残差块
        for block in self.blocks:
            x = block(x)

        # 输出
        output = self.output_layer(x)

        return output


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim, dropout, activation):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播"""
        residual = x
        out = self.layers(x)
        out += residual  # 残差连接
        out = self.act(out)
        out = self.dropout(out)
        return out


class FactorizedMLP(nn.Module):
    """
    因子分解MLP
    使用低秩分解减少参数数量
    """

    def __init__(self, input_dim, hidden_dim, output_dim, rank_ratio=0.5, dropout=0.1):
        """
        Args:
            input_dim: int - 输入维度
            hidden_dim: int - 隐藏层维度
            output_dim: int - 输出维度
            rank_ratio: float - 秩比例 (rank = min(in, out) * rank_ratio)
            dropout: float - dropout概率
        """
        super().__init__()

        # 分解维度
        rank1 = max(1, int(min(input_dim, hidden_dim) * rank_ratio))
        rank2 = max(1, int(min(hidden_dim, output_dim) * rank_ratio))

        # 第一层：输入 -> 低秩 -> 隐藏
        self.down_proj1 = nn.Linear(input_dim, rank1)
        self.up_proj1 = nn.Linear(rank1, hidden_dim)

        # 第二层：隐藏 -> 低秩 -> 输出
        self.down_proj2 = nn.Linear(hidden_dim, rank2)
        self.up_proj2 = nn.Linear(rank2, output_dim)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for proj in [self.down_proj1, self.up_proj1, self.down_proj2, self.up_proj2]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x):
        """前向传播"""
        # 第一层
        x = self.down_proj1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj1(x)
        x = self.dropout(x)

        # 第二层
        x = self.down_proj2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj2(x)

        return x


class GaussianMLP(nn.Module):
    """
    高斯MLP
    输出均值和方差，用于不确定性估计
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: int - 输入维度
            hidden_dim: int - 隐藏层维度
            output_dim: int - 输出维度
            num_layers: int - 层数
            dropout: float - dropout概率
        """
        super().__init__()

        # 共享特征提取器
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        self.feature_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.feature_layers.append(nn.Linear(dims[i], dims[i + 1]))

        # 均值和方差的预测头
        self.mean_head = nn.Linear(output_dim, output_dim)
        self.log_var_head = nn.Linear(output_dim, output_dim)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.feature_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.xavier_uniform_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def forward(self, x):
        """前向传播"""
        # 特征提取
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i < len(self.feature_layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        # 预测均值和方差
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)

        return mean, log_var
