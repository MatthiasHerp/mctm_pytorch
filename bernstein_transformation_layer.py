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
        params_restricted[1:] = torch.matmul(torch.exp(params_a[1:]),torch.triu(torch.ones(degree,degree))) + params_a[1] #abs or exp
        # cannot use exp as it destabilizes the optimisation using LBFGS, get inf for params fast?
        # however with exp works really well and fast with adam
        # Relu works for both, results are good except at lower end there is a cutoff somehow
    else:
        params_restricted = params_a.clone()
    n = degree
    #return sum((params[1] + sum(torch.abs(params[1:(v-1)]))) * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input) for v in range(n+1))

    input_a = (input_a - polynomial_range[0]) / (polynomial_range[1] - polynomial_range[0])

    if derivativ == 0:
        return sum(params_restricted[v] * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input_a) for v in range(n+1)) #before we had: params_restricted[v-1]
    elif derivativ == 1:
        return sum(params_restricted[v] * torch.FloatTensor([n]) * (b(torch.FloatTensor([v-1]), torch.FloatTensor([n-1]), input_a) -
                                    b(torch.FloatTensor([v]), torch.FloatTensor([n-1]), input_a)) for v in range(n+1))

def multivariable_bernstein_prediction(input, degree, number_variables, params, polynomial_range, monotonically_increasing, derivativ=0):
    # input dims: 0: observation number, 1: variable
    # cloning tipp from here: https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/10
    output = input.clone()
    for var_num in range(number_variables):
        output[:,var_num] = bernstein_prediction(params[:,var_num], input[:,var_num], degree, polynomial_range[:,var_num], monotonically_increasing, derivativ)
    return output

class Transformation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range):
        super().__init__()
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = polynomial_range
        # param dims: 0: basis, 1: variable
        p = torch.FloatTensor(np.repeat(np.repeat(1,self.degree+1),self.number_variables))
        self.params = nn.Parameter(torch.reshape(p,(self.degree+1, self.number_variables)))

    def forward(self, input, log_d = 0, inverse = False, monotonically_increasing = True):
        # input dims: 0: observation number, 1: variable
        if not inverse:
            output = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, monotonically_increasing)
            output_first_derivativ = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, monotonically_increasing, derivativ=1)
            log_d = log_d + torch.log(torch.abs(output_first_derivativ)) # Error this is false we require the derivativ of the bernstein polynomial!332'
        else:
            output = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params_inverse, self.polynomial_range_inverse, monotonically_increasing=False)


        return output, log_d

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