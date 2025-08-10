import torch
import torch.nn as nn

__all__ = ["SimAM"]
class SimAM(torch.nn.Module):
    style = "i"
    def __init__(self, placeholder=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

# ----
if __name__ == "__main__":
    model = SimAM(3)
    input_tensor = torch.randn(1, 3, 128, 128)
    output_tensor = model(input_tensor)
    print("input",input_tensor.shape)
    print("output",output_tensor.shape)
# ----
