import torch
from torch import nn


class CKA(nn.Module):
    def __init__(self, model1, model2, dataset1, dataset2=None):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.dataset1 = dataset2
        if not dataset2:
            self.dataset2 = dataset1
        else:
            self.dataset2 = dataset2

    def forward(self, x, y):
        xx = torch.bmm(x.transpose(-1, -2), x)
        yy = torch.bmm(y.transpose(-1, -2), y)
        xy = torch.bmm(y.transpose(-1, -2), x)
        xx_norm = torch.linalg.matrix_norm(xx, ord='fro', dim=(-2, -1))
        yy_norm = torch.linalg.matrix_norm(yy, ord='fro', dim=(-2, -1))
        xy_norm = torch.linalg.matrix_norm(xy, ord='fro', dim=(-2, -1))
        return torch.pow(xy_norm, 2) / (xx_norm * yy_norm)

    def compare(self):
        dataset1 = self.dataset1
        dataset2 = self.dataset2



if __name__ == "__main__":
    cka = CKA(0, 0, 0)

    batch_size = 200
    length = 100
    hidden_dim = 64

    a = torch.randn(size=(batch_size, length, hidden_dim))
    a_linear_transform = torch.matmul(a, torch.randn(size=(hidden_dim, hidden_dim))) + torch.randn(size=(hidden_dim, ))
    a_Q, a_R = torch.qr(a)
    a_linear_transform_Q, a_linear_transform_R = torch.qr(a_linear_transform)
    orthogonal_transform = torch.diag(torch.randint(low=0, high=2, size=(hidden_dim, ))*2 - 1)[torch.randperm(hidden_dim)]
    orthogonal_transform = orthogonal_transform[:, torch.randperm(hidden_dim)].float()
    a_orthogonal_transform = torch.matmul(a, orthogonal_transform)
    a_isotropic_scale = a*torch.randn(1)

    b = torch.exp(torch.rand(size=(batch_size, length, hidden_dim)))
    b_Q, b_R = torch.qr(b)
    b_linear_transform = torch.matmul(b, torch.randn(size=(hidden_dim, hidden_dim))) + torch.randn(size=(hidden_dim,))
    b_linear_transform_Q, b_linear_transform_R = torch.qr(b_linear_transform)
    b_orthogonal_transform = torch.matmul(b, orthogonal_transform)
    b_isotropic_scale = b*torch.randn(1)

    cka_score_similar = cka(a, a)
    cka_score_similar_linear_transform = cka(a, a_linear_transform)
    cka_score_similar_QR = cka(a_Q, a_Q)
    cka_score_similar_QR_linear_transform = cka(a_Q, a_linear_transform_Q)
    cka_score_similar_orthogonal_transform = cka(a, a_orthogonal_transform)
    cka_score_similar_isotropic_scale = cka(a, a_isotropic_scale)

    cka_score_dissimilar = cka(a, b)
    cka_score_dissimilar_linear_transform = cka(a_linear_transform, b_linear_transform)
    cka_score_dissimilar_QR = cka(a_Q, b_Q)
    cka_score_dissimilar_QR_linear_transform = cka(a_linear_transform_Q, b_linear_transform_Q)
    cka_score_dissimilar_orthogonal_transform = cka(a_orthogonal_transform, b_orthogonal_transform)
    cka_score_dissimilar_isotropic_scale = cka(a_isotropic_scale, b_isotropic_scale)


    print("done")

