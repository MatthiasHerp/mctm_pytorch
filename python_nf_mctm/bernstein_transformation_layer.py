import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns
import warnings

from python_nf_mctm.training_helpers import EarlyStopper
from python_nf_mctm.bspline_prediction import bspline_prediction
from python_nf_mctm.bernstein_prediction import bernstein_prediction, compute_multivariate_bernstein_basis, restrict_parameters
from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
from python_nf_mctm.splines_utils import adjust_ploynomial_range
from python_nf_mctm.bspline_prediction import compute_multivariate_bspline_basis


def compute_starting_values(degree, min, max, number_variables, number_covariates=False):
    """
    Computes Starting Values for tge Transformation layer as a line that ranges from the min to the max value of the range
    of the following decorrelation layer.

    :param degree: number of basis functions
    :param min: min of the range
    :param max: max of the range
    :param number_variables: dimensionality of the data, e.g. how many starting value vectors are needed
    :param number_covariates: number of covariates to include (only implemented for 1 for now)
    :return: starting values tensor
    """
    par_restricted_opt = torch.linspace(min,max,degree+1)
    par_unristricted = par_restricted_opt
    par_unristricted[1:] = torch.log(par_restricted_opt[1:] - par_restricted_opt[:-1])#torch.diff(par_restricted_opt[1:]))


    if number_covariates == 1:
        # Only implemented for 1 covariate!
        par_unristricted = par_unristricted.repeat(degree+1,1).T.flatten()
    elif number_covariates > 1:
        raise NotImplementedError("Only implemented for 1 or No covariates!")

    par_restricted_opt = torch.Tensor.repeat(par_unristricted,(number_variables,1)).T
    #par_restricted_opt = torch.reshape(par_restricted_opt,(degree+1,3))

    #a.repeat((4,1)).T.flatten()
    return par_restricted_opt


from statsmodels.distributions.empirical_distribution import ECDF


class Transformation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range, monotonically_increasing=True, spline="bernstein", span_factor=0.1,
                 number_covariates=False, device=None):
        super().__init__()
        self.type = "transformation"
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = polynomial_range
        self.spline = spline
        # param dims: 0: basis, 1: variable
        self.params = nn.Parameter(compute_starting_values(degree,
                                                           polynomial_range[0,0],
                                                           polynomial_range[1,0],
                                                           self.number_variables,
                                                           number_covariates=number_covariates))

        self.monotonically_increasing = monotonically_increasing

        self.span_factor = span_factor

        self.multivariate_bernstein_basis = False
        self.multivariate_bernstein_basis_derivativ_1 = False

        self.number_covariates = number_covariates

        self.device = device

        #self.span_factor_inverse = False
        #self.polynomial_range_inverse = False
        #self.degree_inverse = False
        #self.params_inverse = False

        #if number_covariates is not False:
        #    if number_covariates > 1:
        #        print("Warning, covariates not implemented for more than 1 covariate")
#
        #    #TODO these parameters need to be for a 2D spline model!
        #    #self.params_covariate = nn.Parameter(compute_starting_values_berstein_polynomials(degree,
        #    #                                                                        polynomial_range[0,0],
        #    #                                                                        polynomial_range[1,0],
        #    #                                                                        self.number_variables))
        #    self.params_covariate = compute_starting_values_berstein_polynomials(degree,
        #                                                                         polynomial_range[0,0],
        #                                                                         polynomial_range[1,0],
        #                                                                         self.number_variables)
        #else:
        #    self.params_covariate = False

    def forward(self, input, covariate=False, log_d = 0, inverse = False, return_log_d = False, new_input=True, compute_optimal_initial_params=False):
        # input dims: 0: observaton number, 1: variable
        # Important: set the default of new input to true, otherwise we might use training set for validation results by accident
        #            Thus only specifiy new_input=False during training

        # TODO: add option to redefine the basis (needed for validation step and out of sample prediction as well as sampling)
        #       Done below, should work
        if inverse:
            new_input=True
            warnings.warn("Warning: inverse changes stored basis, set new_input True in next pass through model.")
        # We only want to define the basis once for the entire training
        if new_input is True or self.multivariate_bernstein_basis is False and self.multivariate_bernstein_basis_derivativ_1 is False:
            self.generate_basis(input, covariate, inverse)

        if compute_optimal_initial_params:
            self.compute_initial_parameters_transformation(input, covariate=covariate)

        if not inverse:
            output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = self.transformation(input, derivativ=0)
            output_first_derivativ, _, _, _ = self.transformation(input, derivativ=1)
            log_d = log_d + torch.log(output_first_derivativ) # Error this is false we require the derivativ of the bernstein polynomial!332'
            # took out torch.abs(), misunderstanding, determinant can be a negativ value (flipping of the coordinate system)
        else:
            #TODO: need to pass inverse params
            # maybe put a helper fct into the first line of the transformation fct to choose the forward or backward params
            output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = self.transformation(input, derivativ=0, inverse=True)
        if return_log_d==True:
            return output, log_d
        else:
            return output

    def generate_basis(self, input, covariate, inverse=False):
        if not inverse:
            span_factor = self.span_factor
            polynomial_range = self.polynomial_range
            degree = self.degree
            spline = self.spline
        else:
            span_factor = self.span_factor_inverse
            polynomial_range = self.polynomial_range_inverse
            degree = self.degree_inverse
            spline = self.spline_inverse

        if spline == "bernstein":
            self.multivariate_bernstein_basis = compute_multivariate_bernstein_basis(input=input,
                                                                                     degree=degree,
                                                                                     polynomial_range=polynomial_range,
                                                                                     span_factor=span_factor,
                                                                                     derivativ=0,
                                                                                     covariate=covariate,
                                                                                     device=self.device)

            self.multivariate_bernstein_basis_derivativ_1 = compute_multivariate_bernstein_basis(input=input,
                                                                                                 degree=degree,
                                                                                                 polynomial_range=polynomial_range,
                                                                                                 span_factor=span_factor,
                                                                                                 derivativ=1,
                                                                                                 covariate=covariate,
                                                                                                 device=self.device)
        elif spline == "bspline":
            print("generate basis",input.device)
            self.multivariate_bernstein_basis = compute_multivariate_bspline_basis(input, degree, polynomial_range, span_factor, covariate=False, device=self.device)

            #TODO: need to implement first derivativ basis of bsplines
            self.multivariate_bernstein_basis_derivativ_1 = compute_multivariate_bspline_basis(input, degree, polynomial_range, span_factor, covariate=False, derivativ=1, device=self.device)

    def compute_initial_parameters_transformation(self, input, covariate):
        """
        #### Does Not work see obsidean notes####
        Compute initial parameters for the transformation based on a linear regression

        :param input:
        :param covariate:
        :return:
        """

        # param dims: 0: basis, 1: variable
        if self.number_covariates==False:
            params_tensor = torch.zeros((self.degree+1, self.number_variables), device=self.device)
        else:
            params_tensor = torch.zeros((self.degree+1 + self.number_covariates*(self.degree+1), self.number_variables), device=self.device)

        num_variables = input.size(1)
        for i in range(num_variables):
            y = input[:, i]
            z_true = torch.distributions.Normal(loc=0, scale=1).icdf(torch.tensor(ECDF(y)(y)) - 0.0001).to(self.device)

            #plt.hist(z_true)
            #plt.show()
#
            ## z_true is between 0 and 1
            z_true = (z_true - z_true.min()) / (z_true.max() - z_true.min())
#
            #plt.hist(z_true)
            #plt.show()
#
            ##z_true in the polynomial range
            z_true = z_true * 30 -15 #(self.polynomial_range[1,i] - self.polynomial_range[0,i]) + self.polynomial_range[0,i]
#
            plt.hist(z_true)
            plt.show()



            if covariate == False:
                res = np.linalg.lstsq(self.multivariate_bernstein_basis[:, :, i].detach().numpy(),
                                      z_true.detach().numpy(), rcond=None)
            else:
                expl_variables = torch.vstack([y, covariate])
                res = np.linalg.lstsq(self.multivariate_bernstein_basis[:, :, i].detach().numpy(),
                                      z_true.detach().numpy(), rcond=None)
            param_vec = torch.tensor(res[0])

            #if self.dev is not False:
            #    param_vec.to(self.dev)

            param_vec[0] = -15
            param_vec[param_vec.size(0)-1] = 15
            param_vec = torch.tensor([param.item() if param < 15 else 15 for param in param_vec])

            #if self.dev is not False:
            #    param_vec.to(self.dev)

            param_vec = torch.tensor([param.item() if param > -15 else -15 for param in param_vec])

            #if self.dev is not False:
            #    param_vec.to(self.dev)

            for j in range(1, param_vec.size(0)):
                if param_vec[j] - param_vec[j-1] < 0.001:
                    param_vec[j] = param_vec[j-1] + 0.001
            #param_vec[1:] = torch.tensor([param_vec[i].item() if param_vec[i] - param_vec[i-1] > 0.001 else param_vec[i-1] + 0.001 for i in range(1,param_vec.size(0))])
            differences = torch.diff(param_vec)
            #differences = torch.tensor([diff.item() if diff < torch.log(torch.tensor([4.])).item() else torch.log(torch.tensor([4.])).item() for diff in differences])
            #log_differences = torch.log(torch.tensor([diff.item() if 0 < diff else 0.001 for diff in differences]))
            log_differences = torch.log(differences)
            param_vec = torch.cat([param_vec[0].unsqueeze(0), log_differences])

            params_tensor[:, i] = param_vec.unsqueeze(0)

        self.params = nn.Parameter(params_tensor)

            #from scipy.optimize import minimize
#
            #def objective_function(params, multivariate_bernstein_basis, z_true):
            #    #diff = np.diff(params)
            #    #diff = np.mean([-diff_i if diff_i < 0 else 0 for diff_i in diff])
            #    params_restricted = params
            #    params_restricted[1:] = np.exp(params_restricted[1:])
            #    params_restricted = np.matmul(np.tril(np.ones((params_restricted.shape[0], params_restricted.shape[0]))),
            #                                  params_restricted)
#
            #    preds = np.sum(multivariate_bernstein_basis * params_restricted,1)
            #    #outside_bounds_pen = np.mean([1 if -15 < preds_i < 15 else 0 for preds_i in preds])
#
            #    error = np.mean(np.sum(preds - z_true)) ** 2 #+ 1000000 * diff + 1000000 * outside_bounds_pen
            #    return (error)
#
            #beta_init = np.array([1] * (self.multivariate_bernstein_basis.size()[1]))
            #result = minimize(objective_function, beta_init, args=(self.multivariate_bernstein_basis[:,:,i].numpy(), z_true.numpy()),
            #                  method='BFGS', options={'maxiter': 500})
#
            #param_vec = torch.tensor(result["x"])
            ##differences = torch.diff(param_vec)
            ##differences = torch.tensor([diff.item() if diff > 0 else 0.001 for diff in differences])
            ##log_differences = torch.log(torch.diff(param_vec))

            ##param_vec = torch.cat([param_vec[0].unsqueeze(0),log_differences])

            #params_tensor[:, i] = param_vec.unsqueeze(0)

        #self.params = nn.Parameter(params_tensor)

    def transformation(self, input, derivativ=0, inverse=False):
        # FunFact:
        # due to the basis being compute in generate_basis, the input is only used to have the correct dimensions for the output

        # input dims: 0: observation number, 1: variable
        # cloning tipp from here: https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/10
        #output = input.clone()

        #second_order_ridge_pen_sum = 0
        #first_order_ridge_pen_sum = 0
        # = 0

        if derivativ==0:
            basis = self.multivariate_bernstein_basis.to(input.device)
        elif derivativ==1:
            basis = self.multivariate_bernstein_basis_derivativ_1.to(input.device)

        #if self.spline == "bernstein":
        if not inverse:
            params_restricted = restrict_parameters(params_a=self.params, covariate=self.number_covariates, degree=self.degree, monotonically_increasing=self.monotonically_increasing, device=input.device)
        else:
            params_restricted = restrict_parameters(params_a=self.params_inverse, covariate=self.number_covariates, degree=self.degree_inverse, monotonically_increasing=self.monotonically_increasing_inverse, device=input.device)
        # Explanation:
        # multivariate_bernstein_basis: 0: observation, 1: basis, 2: variable
        # params: 0: basis, 1: variable
        # output: 0: observation, 1: variable
        # Comment: we do normal multiplication as we want to multiply the parameters of each varaible only with its own basis
        #          we sum over dim 1 which is the basis
        #          note we use the restricted parameters
        output = torch.sum(basis * params_restricted.unsqueeze(0), (1))
        #Test: torch.sum(basis * params_restricted.unsqueeze(0), (1))[0,:] == torch.sum(basis[0,:,:] * params_restricted.unsqueeze(0), (1))[0,:]
        #      torch.sum(basis * params_restricted.unsqueeze(0), (1))[1,:] == torch.sum(basis[1,:,:] * params_restricted.unsqueeze(0), (1))[0,:]

        # penalities
        second_order_ridge_pen = 0
        first_order_ridge_pen = 0
        param_ridge_pen = 0

        return output, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen


        # Construction Note: after vectorizing and allowing for 2D tensor basis e.g. covariates this layer is only available using Bernstein Polynomials
        #if self.spline == "bernstein":
        #    output, second_order_ridge_pen_sum, \
        #    first_order_ridge_pen_sum, param_ridge_pen_sum = bernstein_prediction(
        #        multivariate_bernstein_basis=self.multivariate_bernstein_basis
        #        multivariate_bernstein_basis_derivativ_1=self.multivariate_bernstein_basis_derivativ_1,
        #        params_a=self.params,
        #        degree=self.degree,
        #        monotonically_increasing=self.monotonically_increasing,
        #        derivativ=derivativ,
        #        covariate=self.number_covariates)


        #for var_num in range(self.number_variables):
#
        #    if self.spline == "bernstein":
        #        output[:, var_num], second_order_ridge_pen_current, \
        #        first_order_ridge_pen_current, param_ridge_pen_current = bernstein_prediction(params_a=self.params[:, var_num],
        #                                                                                      input_a=input[:, var_num],
        #                                                                                      degree=self.degree,
        #                                                                                      polynomial_range=self.polynomial_range[:, var_num],
        #                                                                                      monotonically_increasing=self.monotonically_increasing,
        #                                                                                      derivativ=derivativ,
        #                                                                                      span_factor=self.span_factor,
        #                                                                                      covariate=self.covariate)
        #    elif self.spline == "bspline":
        #        # TODO: bsplines are not implemented for the 2D tensor case yet
        #        output[:, var_num], second_order_ridge_pen_current, \
        #        first_order_ridge_pen_current, param_ridge_pen_current = bspline_prediction(params_a=self.params[:, var_num],
        #                                                                                    input_a=input[:, var_num],
        #                                                                                    degree=self.degree,
        #                                                                                    polynomial_range=self.polynomial_range[:,var_num],
        #                                                                                    monotonically_increasing=self.monotonically_increasing,
        #                                                                                    derivativ=derivativ,
        #                                                                                    span_factor=self.span_factor,
        #                                                                                    covariate=self.covariate,
        #                                                                                    return_penalties=True)
#
        #    second_order_ridge_pen_sum += second_order_ridge_pen_current
        #    first_order_ridge_pen_sum += first_order_ridge_pen_current
        #    param_ridge_pen_sum += param_ridge_pen_current

        #return output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum


    def approximate_inverse(self, input, monotonically_increasing_inverse=False, spline_inverse="bspline", degree_inverse=150, iterations=1000, lr=1, patience=20, min_delta=1e-4, global_min_loss=0.001, span_factor_inverse=0.2,
                            input_covariate=False):
        # optimization using linespace data and the forward berstein polynomial?

        if input_covariate is not False:
            covariate_space = input_covariate[input_covariate.multinomial(100000, replacement=True)]

        if degree_inverse == 0:
            degree_inverse = 2 * self.degree

        #self.monotonically_increasing_inverse = monotonically_increasing_inverse
        self.span_factor_inverse = span_factor_inverse

        #a, b = torch.meshgrid([torch.linspace(input[:,0].min(),input[:,0].max(),100),torch.linspace(input[:,1].min(),input[:,1].max(),100)])
        #input_space = torch.vstack([a.flatten(),b.flatten()]).T

        input_space = torch.zeros((100000, self.number_variables), dtype=torch.float32)
        for var_number in range(self.number_variables):
            input_space[:, var_number] = torch.linspace(input[:,var_number].min(),input[:,var_number].max(),100000,device=input.device)

        #input_space = torch.vstack([torch.linspace(input[:,0].min(),input[:,0].max(),10000),
        #                            torch.linspace(input[:,1].min(),input[:,1].max(),10000)]).T

        if input_covariate is not False:
            output_space = self.forward(input_space,covariate_space)
        else:
            output_space = self.forward(input_space)

        polynomial_range_inverse = torch.zeros((2, self.number_variables), dtype=torch.float32, device=input.device)

        for var_number in range(self.number_variables):
            span_var_number = output_space[:, var_number].max() - output_space[:, var_number].min()
            polynomial_range_inverse[:, var_number] = torch.tensor([output_space[:, var_number].min() - span_var_number*span_factor_inverse,
                                                                    output_space[:, var_number].max() + span_var_number*span_factor_inverse],
                                                                   dtype=torch.float32, device=input.device)

        #span_0 = output_space[:, 0].max() - output_space[:, 0].min()
        #span_1 = output_space[:, 1].max() - output_space[:, 1].min()
        #polynomial_range_inverse = torch.tensor([[output_space[:, 0].min() - span_0*span_factor_inverse, output_space[:, 1].min() - span_1*span_factor_inverse],
        #                                         [output_space[:, 0].max() + span_0*span_factor_inverse, output_space[:, 1].max() + span_1*span_factor_inverse]], dtype=torch.float32)

        #input_space = input
        #output_space = multivariable_bernstein_prediction(input_space, self.degree, self.number_variables, self.params, monotonically_increasing=True)

        self.monotonically_increasing_inverse = False #due to ols estimation below

        inv_trans = Transformation(degree=degree_inverse,
                                   number_variables=self.number_variables,
                                   polynomial_range=polynomial_range_inverse,
                                   monotonically_increasing=self.monotonically_increasing_inverse,
                                   spline=spline_inverse,
                                   number_covariates=self.number_covariates)

        inv_trans.generate_basis(input=output_space.detach(),covariate=False,inverse=False)

        params_tensor = inv_trans.params.clone()
        for num_var in range(inv_trans.params.size(1)):
            res = np.linalg.lstsq(inv_trans.multivariate_bernstein_basis[:,:,num_var].detach().numpy(), input_space[:,num_var].detach().numpy(), rcond=None)
            params_tensor[:,num_var] = torch.tensor(res[0], dtype=torch.float32)

        self.polynomial_range_inverse = polynomial_range_inverse
        self.params_inverse = nn.Parameter(params_tensor)
        self.spline_inverse = spline_inverse
        self.degree_inverse = degree_inverse

        ##def se(y_estimated, y_train):
        ##    return torch.sum((y_train - y_estimated)**2)
#
        ##loss_mse = se()
        #opt = FullBatchLBFGS(inv_trans.parameters(), lr=lr, history_size=1, line_search="Wolfe")
#
        #loss_fct = nn.L1Loss()  # MSELoss L1Loss
        #early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)
#
#
        #def closure():
        #    opt.zero_grad()
        #    # new_input = False as we define a new layer to train the inverse hence it will use its first input to create basis
        #    if input_covariate is not False:
        #        input_space_pred = inv_trans.forward(output_space.detach(),covariate=covariate_space.detach(),new_input=False)
        #    else:
        #        input_space_pred = inv_trans.forward(output_space.detach(), new_input=False)
        #    loss = loss_fct(input_space_pred, input_space.detach())  # use the `objective` function
        #    #loss.backward(retain_graph=True)  # backpropagate the loss
        #    return loss
#
        #loss = closure()
        #loss.backward()
        #options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
#
        ##opt_inv  = optim.Adam(inv_trans.parameters(), lr = lr, weight_decay=weight_decay)
        ##scheduler_inv = optim.lr_scheduler.StepLR(opt_inv, step_size = 500, gamma = 0.5)
        #loss_list = []
#
        #for i in tqdm(range(iterations)):
#
        #    # needs to be computed manually at each step
        #    #input_space_comp = input_space
        #    #output_space, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_bernstein_prediction(input_space_comp, self.degree, self.number_variables, self.params, self.polynomial_range, monotonically_increasing=True)
#
        #    #opt_inv.zero_grad() # zero out gradients first on the optimizer
        #    #input_space_pred = inv_trans.forward(output_space.detach())
        #    #current_loss = loss_fct(input_space_pred, input_space.detach())
        #    #l2_losses.append(current_loss.detach().numpy())
#
        #    #current_loss = loss_mse(input_space_pred, input_space_comp) # use the `objective` function
#
        #    #current_loss.backward() # backpropagate the loss
        #    #opt_inv.step()
        #    #scheduler_inv.step()
#
        #    #opt.step(closure)
#
        #    current_loss, _, _, _, _, _, _, _ = opt.step(options)
        #    loss_list.append(current_loss.detach().numpy().item())
#
        #    if early_stopper.early_stop(current_loss=current_loss.detach().numpy(), model=inv_trans):
        #        print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience,
        #              "and min_delta", min_delta)
        #        break
#
        ## Return the best model which is not necessarily the last model
        #inv_trans = Transformation(degree=degree_inverse,
        #                           number_variables=self.number_variables,
        #                           polynomial_range=polynomial_range_inverse,
        #                           monotonically_increasing=monotonically_increasing_inverse,
        #                           spline=spline_inverse,
        #                           number_covariates=self.number_covariates)
#
        #inv_trans.load_state_dict(early_stopper.best_model_state)
#
        ##input_space_pred_final = inv_trans.forward(output_space.detach())
        ##loss_final = loss_fct(input_space_pred_final, input_space.detach())
#
#
        #print("Final loss", early_stopper.min_loss)
#
        ## Plot neg_log_likelihoods over training iterations:
        #fig, ax = plt.subplots(figsize=(6, 6))
        #sns.lineplot(data=loss_list, ax=ax)
        #plt.xlabel("Iteration")
        #plt.ylabel("Loss")
#
        #self.polynomial_range_inverse = polynomial_range_inverse
        #self.params_inverse = inv_trans.params
        #self.spline_inverse = spline_inverse
        #self.degree_inverse = degree_inverse
#
        #return fig

    #TODO: repr needs to be redone
    def __repr__(self):
        return "Transformation(degree={degree:.2f}, params={params:.2f})".format(degree = self.degree, params = self.params)