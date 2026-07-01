import torch
import torch.nn as nn
import torch.nn.functional as F


class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()


class LinearClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale


class SeparatedLearnableClassifier(_Classifier):
    def __init__(
            self,
            feat_dim: int,
            num_classes: int,
            dtype=None,
            scale: float = 30.0,
            delta_scale: float = 0.01,
            **kwargs
    ):
        super().__init__(feat_dim, num_classes, dtype)

        # Residual transformation: T = I + eps * Delta
        self.delta = nn.Parameter(torch.zeros(feat_dim, feat_dim, dtype=dtype))
        self.delta_scale = delta_scale

        self.scale = scale
        self.feat_dim = feat_dim

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        T = torch.eye(
            self.feat_dim,
            device=x.device,
            dtype=x.dtype
        ) + self.delta_scale * self.delta

        w = self.weight @ T.T
        w = F.normalize(w, dim=-1)

        logits = F.linear(x, w) * self.scale
        return logits


class CLIPTextWeightedClassifier(_Classifier):
    def __init__(
            self,
            feat_dim: int = None,
            num_classes: int = None,
            dtype=None,
            scale: float = 30.0,
            **kwargs
    ):
        # 初始化父类，此时 weight 占位符形状可能不正确，稍后通过 apply_weight 修正
        super().__init__(feat_dim, num_classes, dtype)

        self.scale = scale
        self.feat_dim = feat_dim
        # text_weights (W) 将在 apply_weight 中根据实际输入的文本特征维度动态初始化
        self.text_weights = None

    def apply_weight(self, weight):
        """
        设置文本特征权重。
        参数:
            weight: torch.Tensor, 形状为 [C, M, feat_dim]
                    C: 类别数量 (num_classes)
                    M: 每个类别的文本描述数量 (自动推导)
                    feat_dim: 特征维度
        """
        if weight.dim() != 3:
            raise ValueError(f"Expected 3D tensor [C, M, feat_dim], got {weight.dim()}D")

        c_dim, m_dim, f_dim = weight.shape

        if c_dim != self.num_classes:
            raise ValueError(f"Number of classes in weight ({c_dim}) does not match num_classes ({self.num_classes})")
        if f_dim != self.feat_dim:
            raise ValueError(f"Feature dim in weight ({f_dim}) does not match feat_dim ({self.feat_dim})")

        # 更新父类的 weight 为展平后的形式或者保留原状供参考，这里为了符合父类逻辑，
        # 我们主要将详细的 [C, M, D] 存储在内部变量中，父类的 self.weight 可作为中心向量或忽略
        # 为了兼容父类结构，我们将 [C, M, D]  reshape 为 [C, M*D] 存储或者单独存储
        # 这里选择单独存储详细特征，父类的 self.weight 可以初始化为均值或其他，但不参与核心计算逻辑

        # 保存完整的文本特征 [C, M, feat_dim]
        self.text_embeddings = weight.clone().to(dtype=self.dtype)

        # 初始化可学习的重要性矩阵 W [C, M]
        # 初始化为均匀分布，表示初始时每条文本重要性相同
        self.text_weights = nn.Parameter(torch.ones(c_dim, m_dim, dtype=self.dtype))

        # 可选：更新父类的 weight 为文本特征的均值，以便其他通用方法调用时不报错
        self.weight.data = self.text_embeddings.mean(dim=1).clone()

    def forward(self, x):
        """
        计算视觉特征与加权文本特征的相似度。
        参数:
            x: torch.Tensor, 视觉特征 [batch_size, feat_dim]
        返回:
            logits: torch.Tensor, [batch_size, num_classes]
        """
        if self.text_embeddings is None or self.text_weights is None:
            raise RuntimeError("Text embeddings and weights not initialized. Call apply_weight first.")

        batch_size = x.shape[0]
        c_dim, m_dim, _ = self.text_embeddings.shape

        # 1. 归一化视觉特征 [batch, feat_dim]
        x_norm = F.normalize(x, dim=-1)

        # 2. 归一化文本特征 [C, M, feat_dim] -> [C, M, feat_dim]
        text_norm = F.normalize(self.text_embeddings, dim=-1)

        # 3. 计算相似度
        # x_norm: [B, D], text_norm: [C, M, D]
        # 结果 similarity: [B, C, M]
        # 使用 einsum 高效计算: b,d * c,m,d -> b,c,m
        similarity = torch.einsum('bd,cmd->bcm', x_norm, text_norm)

        # 4. 获取权重并进行 Softmax 归一化 (沿 M 维度)
        # self.text_weights: [C, M]
        # 我们希望每个类别下，M 条文本的权重和为 1
        weights_soft = F.softmax(self.text_weights, dim=1)  # [C, M]

        # 5. 加权平均
        # similarity: [B, C, M], weights_soft: [C, M] -> 需要广播
        # 结果 weighted_logits: [B, C]
        weighted_logits = torch.einsum('bcm,cm->bc', similarity, weights_soft)

        return weighted_logits * self.scale


class L2NormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)

    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class MahalanobisClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.num_classes = num_classes
        # 为每个类别维护一个协方差矩阵的逆矩阵
        self.register_buffer('cov_inv', torch.eye(feat_dim, dtype=dtype).unsqueeze(0).repeat(num_classes, 1, 1))
        # 类别计数器，用于增量更新协方差矩阵
        self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.long))

    def forward(self, x):
        """
        计算输入特征到各类别中心的马氏距离
        x: [batch_size, feat_dim]
        返回: [batch_size, num_classes] 距离值（负数表示，数值越大越接近）
        """
        batch_size = x.shape[0]
        distances = torch.zeros(batch_size, self.num_classes, dtype=x.dtype, device=x.device)

        for i in range(self.num_classes):
            # 计算到第i个类别中心的马氏距离
            diff = x - self.weight[i]  # [batch_size, feat_dim]
            # 马氏距离公式: sqrt((x-μ)^T Σ^-1 (x-μ))
            # 但我们返回负的距离值以便于优化（较大的值表示更接近）
            mahalanobis_dist_squared = torch.sum((diff @ self.cov_inv[i]) * diff, dim=1)
            distances[:, i] = -torch.sqrt(mahalanobis_dist_squared + 1e-8)

        return distances

    def update_covariance(self, x, labels):
        """
        根据新样本更新各类别的协方差矩阵
        x: [batch_size, feat_dim]
        labels: [batch_size] 类别标签
        """
        with torch.no_grad():
            for i in range(self.num_classes):
                # 获取属于类别i的样本
                mask = (labels == i)
                if mask.sum() > 0:
                    class_samples = x[mask]
                    class_mean = class_samples.mean(dim=0)

                    # 更新类别中心（移动平均）
                    self.weight[i] = (self.weight[i] * self.class_counts[i] +
                                      class_samples.sum(dim=0)) / (self.class_counts[i] + mask.sum())

                    # 更新协方差矩阵估计
                    if class_samples.shape[0] > 1:
                        # 计算新的协方差矩阵
                        diff = class_samples - class_mean
                        new_cov = (diff.t() @ diff) / (class_samples.shape[0] - 1)

                        # 增量更新协方差矩阵的逆
                        # 简化的更新方式：使用新估计值
                        reg_new_cov = new_cov + 1e-6 * torch.eye(self.weight.shape[1],
                                                                 dtype=new_cov.dtype,
                                                                 device=new_cov.device)
                        self.cov_inv[i] = torch.inverse(reg_new_cov)

                    # 更新计数器
                    self.class_counts[i] += mask.sum()

