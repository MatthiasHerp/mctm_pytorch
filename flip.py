import torch
from torch import nn

class Flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tuple, inverse):
        input, log_d_global, param_ridge_pen_global, first_order_ridge_pen_global, second_order_ridge_pen_global = input_tuple
        output_tuple = (torch.flip(input, [0]), log_d_global, param_ridge_pen_global, first_order_ridge_pen_global, second_order_ridge_pen_global)

        return output_tuple