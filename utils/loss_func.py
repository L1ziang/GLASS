import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = [
    'Attention_Loss', 
    'NoPeek_Loss',
    'MI_Loss',
    'siamese_Loss',
    'cloak_Loss',
    'disco_Loss',
]


class Attention_Loss(torch.nn.modules.loss._Loss):

    def __init__(self, maps_type = 'sum_pow', pow = 2):
        super().__init__()
        type_choices = ['sum','sum_pow','max_pow']
        if maps_type not in type_choices:
            raise Exception("unkown maps type.valid choices are {}".format(str(type_choices)))
        else:
            self.maps = maps_type
        self.p = pow

    def forward(self, input_feat, target_feat):
        input_attention = self._attention(input_feat).view(-1)
        target_attention = self._attention(target_feat).view(-1)
        input_attention_norm = torch.norm(input_attention)
        target_attention_norm = torch.norm(target_attention)

        # Put it together
        attention_loss = torch.norm(input_attention/input_attention_norm - target_attention/target_attention_norm)
        return attention_loss

    def _sum(self, feat):
        return torch.sum(feat.abs(), dim=1)

    def _sum_power_p(self, feat):
        return torch.sum(feat.abs().pow(self.p), dim=1)

    def _max_power_p(self, feat):
        return torch.max(feat.abs().pow(self.p), dim=1)[0]

    def _attention(self, feat):
        if self.maps == 'sum':
            return self._sum(feat)
        elif self.maps == 'sum_pow':
            return self._sum_power_p(feat)
        elif self.maps == 'max_pow':
            return self._max_power_p(feat)


class NoPeek_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, dcor_weighting: float = 0.1) -> None:
        super().__init__()
        self.dcor_weighting = dcor_weighting

        # self.ce = torch.nn.CrossEntropyLoss()
        self.dcor = DistanceCorrelationLoss()

    def forward(self, inputs, intermediates, outputs, targets):
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # DistanceCorrelationLoss is costly, so only calc it if necessary
        if self.dcor_weighting > 0.0:
            loss += self.dcor_weighting * self.dcor(inputs, intermediates)

        return loss
    
class disco_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, dcor_weighting: float = 0.1) -> None:
        super().__init__()
        self.dcor_weighting = dcor_weighting

        # self.ce = torch.nn.CrossEntropyLoss()
        # self.dcor = DistanceCorrelationLoss()

    def forward(self, inputs, noised_intermediates, outputs, targets):
        loss1 = F.binary_cross_entropy_with_logits(outputs, targets)
        # DistanceCorrelationLoss is costly, so only calc it if necessary
        norm_noised_intermidiate = noised_intermediates.norm()

        loss2 = self.dcor_weighting * norm_noised_intermidiate

        if self.dcor_weighting > 0.0:
            loss = loss1 + loss2
        # print("acc loss : {}, norm loss : {}".format(loss1.item(), loss2.item()))
        

        return loss
    
class cloak_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, dcor_weighting: float = 0.1) -> None:
        super().__init__()
        self.dcor_weighting = dcor_weighting

        # self.ce = torch.nn.CrossEntropyLoss()
        # self.dcor = DistanceCorrelationLoss()

    def forward(self, inputs, noised_intermediates, outputs, targets):
        loss1 = F.binary_cross_entropy_with_logits(outputs, targets)
        # DistanceCorrelationLoss is costly, so only calc it if necessary
        norm_noised_intermidiate = noised_intermediates.norm()

        loss2 = self.dcor_weighting * norm_noised_intermidiate

        if self.dcor_weighting > 0.0:
            loss = loss1 - loss2
        # print("acc loss : {}, norm loss : {}".format(loss1.item(), loss2.item()))
        

        return loss
    
class siamese_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, dcor_weighting: float = 0.1) -> None:
        super().__init__()
        self.margin = 30
        self.dcor_weighting = dcor_weighting

    def get_mask(self, labels):
        labels = labels.unsqueeze(-1).to(dtype=torch.float64)
        class_diff = torch.cdist(labels, labels, p=1.0)
        return torch.clamp(class_diff, 0, 1)

    def get_pairwise(self, z):
        z = z.view(z.shape[0], -1)
        return torch.cdist(z, z, p=2.0)

    def forward(self, inputs, intermediates, outputs, targets):
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # DistanceCorrelationLoss is costly, so only calc it if necessary
        mask = self.get_mask(targets).cuda()
        pairwise_dist = self.get_pairwise(intermediates)
        # print(pairwise_dist.mean())
        loss1 = (1 - mask) * pairwise_dist +\
               mask * torch.maximum(torch.tensor(0.).cuda(), self.margin - pairwise_dist)
        
        # print("loss  :", loss)
        # print("loss1 :", loss1.mean())

        if self.dcor_weighting > 0.0:
            loss += self.dcor_weighting * loss1.mean()

        return loss

# def entropy(x, k=100, sigma=None):
#     """
#     计算x的熵
#     :param x: tensor，shape为(N, *)，N为样本数，*表示任意维度
#     :param k: int，k近邻的数量
#     :param sigma: float，高斯核的标准差，如果为None，则使用k近邻的距离的中位数作为标准差
#     :return: float，熵的值
#     """
#     n = x.shape[0]
#     # print(n)
#     distances = torch.cdist(x, x) # 计算距离
#     distances, topk_indices = torch.topk(distances, k=k+1, largest=False, dim=1)
#     # topk_distances = topk_distances.tolist()  # 转换为 Python 列表
#     # topk_indices = topk_indices.tolist()  # 转换为 Python 列表

#     # 计算k近邻距离
#     # nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(x)
#     # distances, indices = nbrs.kneighbors(x)
#     # print(distances.shape)

#     # 计算局部方差
#     if sigma is None:
#         sigma = torch.median(distances[:, 1:])
#     var = (sigma ** 2) * x.shape[1]

#     # 计算熵
#     # print(var)
#     if var < 0:
#         var = torch.Tensor([-1.0]).cuda()
#     const = torch.log(2 * torch.pi * var) * 0.5 +1e-3
#     entropy = torch.sum(torch.log(distances[:, 1:]) + const) / n

#     # print(const)

#     return entropy

# def tensor_entropy(tensor):
#     # 计算张量T中每个元素的对数
#     log_tensor = torch.log(tensor)
#     # print(log_tensor)
#     # 将T中为0的元素对应的对数替换为0，避免后续计算出现NaN
#     log_tensor[torch.isinf(log_tensor)] = 1e+5
#     log_tensor[torch.isnan(log_tensor)] = 1e-5
#     # 计算张量的熵
#     entropy = -torch.sum(tensor * log_tensor)
#     return entropy

# def mutual_information(x, y, k=60, sigma=None):
#     """
#     估计x和y之间的互信息
#     :param x: tensor，shape为(N, *)，N为样本数，*表示任意维度
#     :param y: tensor，shape为(N, *)，N为样本数，*表示任意维度
#     :param k: int，k近邻的数量
#     :param sigma: float，高斯核的标准差，如果为None，则使用k近邻的距离的中位数作为标准差
#     :return: float，互信息的值
#     """
#     # 将x和y连接成一个tensor
#     xy = torch.cat([x, y], dim=1)

#     # 计算熵
#     hx = tensor_entropy(x)
#     hy = tensor_entropy(y)
#     hxy = tensor_entropy(xy)

#     # 计算互信息
#     mi = hx + hy - hxy
#     print(mi)

#     return mi

def KL_between_normals(mu_q, sigma_q, mu_p, sigma_p):
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    # print(mu_diff_sq.shape)
    # print((sigma_p**2).shape)

    fs = torch.sum(torch.div(sigma_q**2, sigma_p**2), dim=1) + torch.sum(
        torch.div(mu_diff_sq, sigma_p**2), dim=1
    )
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5
    
class MI_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, dcor_weighting: float = 0.1) -> None:
        super().__init__()
        self.dcor_weighting = dcor_weighting

        # self.ce = torch.nn.CrossEntropyLoss()
        # self.dcor = DistanceCorrelationLoss()
        

    def forward(self, inputs, intermediates, outputs, targets):
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # DistanceCorrelationLoss is costly, so only calc it if necessary
        if self.dcor_weighting > 0.0:
            # print(inputs.shape[0])
            # exit()
            intermediates = intermediates.view(intermediates.size(0), -1)
            p_z_given_x_mu = intermediates[:, : int(intermediates.size(1)/2)]
            p_z_given_x_sigma = torch.nn.functional.softplus(intermediates[:, int(intermediates.size(1)/2) :])

            approximated_z_mean = torch.zeros_like(p_z_given_x_mu)
            approximated_z_sigma = torch.ones_like(p_z_given_x_sigma)
            I_ZX_bound = torch.mean(KL_between_normals(p_z_given_x_mu, p_z_given_x_sigma, approximated_z_mean, approximated_z_sigma))
            loss += self.dcor_weighting * I_ZX_bound

        return loss


class DistanceCorrelationLoss(torch.nn.modules.loss._Loss):
    def forward(self, input_data, intermediate_data):
        input_data = input_data.view(input_data.size(0), -1)
        intermediate_data = intermediate_data.view(intermediate_data.size(0), -1)

        # Get A matrices of data
        A_input = self._A_matrix(input_data)
        A_intermediate = self._A_matrix(intermediate_data)

        # Get distance variances
        input_dvar = self._distance_variance(A_input)
        intermediate_dvar = self._distance_variance(A_intermediate)

        # Get distance covariance
        dcov = self._distance_covariance(A_input, A_intermediate)

        # Put it together
        dcorr = dcov / (input_dvar * intermediate_dvar).sqrt()

        return dcorr

    def _distance_covariance(self, a_matrix, b_matrix):
        return (a_matrix * b_matrix).sum().sqrt() / a_matrix.size(0)

    def _distance_variance(self, a_matrix):
        return (a_matrix ** 2).sum().sqrt() / a_matrix.size(0)

    def _A_matrix(self, data):
        distance_matrix = self._distance_matrix(data)

        row_mean = distance_matrix.mean(dim=0, keepdim=True)
        col_mean = distance_matrix.mean(dim=1, keepdim=True)
        data_mean = distance_matrix.mean()

        return distance_matrix - row_mean - col_mean + data_mean

    def _distance_matrix(self, data):
        # n = data.size(0)
        # distance_matrix = torch.zeros((n, n)).to(data.device)

        # for i in range(n):
        #     for j in range(n):
        #         row_diff = data[i] - data[j]
        #         distance_matrix[i, j] = (row_diff ** 2).sum()

        # return distance_matrix

        n = data.size(0)
        G = torch.matmul(data, data.T)
        H = torch.diag(G).repeat((n, 1))
        distance_matrix = H + H.T - 2*G
        return distance_matrix