import torch


__all__ = [
    'Attention_Loss', 
    'NoPeek_Loss',
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

        self.ce = torch.nn.CrossEntropyLoss()
        self.dcor = DistanceCorrelationLoss()

    def forward(self, inputs, intermediates, outputs, targets):
        loss = self.ce(outputs, targets)

        # DistanceCorrelationLoss is costly, so only calc it if necessary
        if self.dcor_weighting > 0.0:
            loss += self.dcor_weighting * self.dcor(inputs, intermediates)

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