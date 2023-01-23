import torch
from torch import nn
import numpy as np
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns

# https://github.com/pytorch/pytorch/issues/47841
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html
def torch_binom(n, v):
    #mask = n.detach() >= v.detach()
    #n = mask * n
    #v = mask * v
    a = torch.lgamma(n + 1) - torch.lgamma((n - v) + 1) - torch.lgamma(v + 1)
    return torch.exp(a)# * mask

# https://en.wikipedia.org/wiki/Bernstein_polynomial
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.html
def b(v, n, x):
    return torch_binom(n, v) * x**v * (1 - x)**(n - v)

def bernstein_prediction(params_a, input_a, degree, polynomial_range, monotonically_increasing=False, derivativ=0):
    # Restricts Monotonically increasing my insuring that params increase
    if monotonically_increasing:

        params_restricted = params_a.clone()
        #relu_obj = nn.ReLU()
        params_restricted[1:] = torch.matmul(torch.exp(params_a[1:]),torch.triu(torch.ones(degree,degree))) + params_a[0] #abs or exp
        #.double() otherwise I started getting the folowing error:
        #{RuntimeError}Expected object of scalar type Double but got scalar type Float for argument #2 'mat2' in call to _th_mm
        #alternativ set dtype=torch.float64 in torch.ones(degree,degree)

        # cannot use exp as it destabilizes the optimisation using LBFGS, get inf for params fast?
        # however with exp works really well and fast with adam
        # Relu works for both, results are good except at lower end there is a cutoff somehow
    else:
        params_restricted = params_a.clone()
    n = degree
    #return sum((params[1] + sum(torch.abs(params[1:(v-1)]))) * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input) for v in range(n+1))

    # penalities
    second_order_ridge_pen = 0
    first_order_ridge_pen = 0
    param_ridge_pen = 0

    input_a = (input_a - polynomial_range[0]) / (polynomial_range[1] - polynomial_range[0])

    if derivativ == 0:
        output = sum(params_restricted[v] * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input_a) for v in range(n+1)) #before we had: params_restricted[v-1]

        return output, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
        #return (output + polynomial_range[0]) / (polynomial_range[1] - polynomial_range[0])

    elif derivativ == 1:
        output = sum(params_restricted[v] * torch.FloatTensor([n]) * (b(torch.FloatTensor([v-1]), torch.FloatTensor([n-1]), input_a) -
                                    b(torch.FloatTensor([v]), torch.FloatTensor([n-1]), input_a)) for v in range(n+1))

        return output, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
        #return (output + polynomial_range[0]) / (polynomial_range[1] - polynomial_range[0])

def multivariable_bernstein_prediction(input, degree, number_variables, params, polynomial_range, monotonically_increasing, derivativ=0):
    # input dims: 0: observation number, 1: variable
    # cloning tipp from here: https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/10
    output = input.clone()

    second_order_ridge_pen_sum = 0
    first_order_ridge_pen_sum = 0
    param_ridge_pen_sum = 0

    for var_num in range(number_variables):
        output[:,var_num], second_order_ridge_pen_current, \
        first_order_ridge_pen_current, param_ridge_pen_current = bernstein_prediction(params[:,var_num], input[:,var_num], degree, polynomial_range[:,var_num], monotonically_increasing, derivativ)

        second_order_ridge_pen_sum += second_order_ridge_pen_current
        first_order_ridge_pen_sum += first_order_ridge_pen_current
        param_ridge_pen_sum += param_ridge_pen_current

    return output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum

def compute_starting_values_berstein_polynomials(degree,min,max):
    par_restricted_opt = torch.tensor(np.linspace(min,max,degree+1), dtype=torch.float32)
    par_unristricted = par_restricted_opt
    par_unristricted[1:] = torch.log(par_restricted_opt[1:] - par_restricted_opt[:-1])#torch.diff(par_restricted_opt[1:]))

    par_restricted_opt = torch.Tensor.repeat(par_unristricted,(3,1)).T
    #par_restricted_opt = torch.reshape(par_restricted_opt,(degree+1,3))

    return par_restricted_opt

class Transformation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range):
        super().__init__()
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = polynomial_range
        # param dims: 0: basis, 1: variable
        self.params = nn.Parameter(compute_starting_values_berstein_polynomials(10,
                                                                                polynomial_range[0,0],
                                                                                polynomial_range[1,0]))

    def forward(self, input_tuple, inverse, log_d = 0, monotonically_increasing = True, return_log_d = False):

        input, log_d_global, param_ridge_pen_global, first_order_ridge_pen_global, second_order_ridge_pen_global = input_tuple

        # input dims: 0: observaton number, 1: variable
        if not inverse:
            output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, monotonically_increasing)
            # Computing derivative here the outputed penalty terms are a nuisance (not required)
            output_first_derivativ, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, monotonically_increasing, derivativ=1)
            log_d = log_d + torch.log(torch.abs(output_first_derivativ)) # Error this is false we require the derivativ of the bernstein polynomial!332'
        else:
            output = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params_inverse, self.polynomial_range_inverse, monotonically_increasing=False)
            log_d = 0
            param_ridge_pen_sum = 0
            first_order_ridge_pen_sum = 0
            second_order_ridge_pen_sum = 0

        log_d_global += log_d
        param_ridge_pen_global += param_ridge_pen_sum
        first_order_ridge_pen_global += first_order_ridge_pen_sum
        second_order_ridge_pen_global += second_order_ridge_pen_sum

        return (output, log_d_global, param_ridge_pen_global, first_order_ridge_pen_global, second_order_ridge_pen_global)


    def approximate_inverse(self, input, polynomial_range_inverse, iterations=4000):
        # optimization using linespace data and the forward berstein polynomial?

        self.polynomial_range_inverse = polynomial_range_inverse

        #input_space = input
        #output_space = multivariable_bernstein_prediction(input_space, self.degree, self.number_variables, self.params, monotonically_increasing=True)

        inv_trans = Transformation(self.degree, self.number_variables, self.polynomial_range_inverse)
        loss_mse = nn.MSELoss()
        opt_inv  = optim.Adam(inv_trans.parameters(), lr = 1e-2)
        scheduler_inv = optim.lr_scheduler.StepLR(opt_inv, step_size = 500, gamma = 0.8)
        l2_losses = []
        for _ in tqdm(range(iterations)):

            # needs to be computed manually at each step
            input_space = input
            output_space = multivariable_bernstein_prediction(input_space, self.degree, self.number_variables, self.params, self.polynomial_range, monotonically_increasing=True)

            opt_inv.zero_grad() # zero out gradients first on the optimizer
            input_space_pred, input_space_pred_log_d  = inv_trans.forward(output_space, inverse=False, monotonically_increasing=False)

            l2_loss = loss_mse(input_space_pred, input_space) # use the `objective` function

            l2_loss.backward() # backpropagate the loss
            opt_inv.step()
            scheduler_inv.step()
            l2_losses.append(l2_loss.detach().numpy())

        self.params_inverse = inv_trans.params

        with sns.axes_style('ticks'):
            plt.plot(l2_losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
        sns.despine(trim = True)


    def __repr__(self):
        return "Transformation(degree={degree:.2f}, params={params:.2f})".format(degree = self.degree, params = self.params)