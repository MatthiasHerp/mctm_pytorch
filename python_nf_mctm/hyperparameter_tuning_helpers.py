import torch
import itertools
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from python_nf_mctm.nf_mctm import *
from python_nf_mctm.training_helpers import *
import optuna
from optuna.samplers import TPESampler

#TODO: y_validate determined here so no need to pass it in
def run_hyperparameter_tuning(y_train: torch.Tensor,
                              y_validate: torch.Tensor,
                              poly_span_abs: float,
                              iterations: int,
                              spline_decorrelation: str,
                              penvalueridge_list: list,
                              penfirstridge_list: list,
                              pensecondridge_list: list,
                              lambda_penalty_params_list: list,
                              learning_rate_list: list,
                              patience_list: list,
                              min_delta_list: list,
                              degree_transformations_list: list,
                              degree_decorrelation_list: list,
                              x_train: torch.Tensor = False,
                              x_validate: torch.Tensor = False,
                              tuning_mode="optuna",
                              cross_validation_folds=False,
                              tune_precision_matrix_penalty=False,
                              device=None):
                          #normalisation_layer_list: list):
    """
    Generates List of all combinations of hyperparameter values from lists
    For each combination does a 5-fold cross validation and trains the model
    It then computes the log likelihood of the model on the validation set and
    stores it all in aa pandas dataframe

    :param model:
    :param y:
    :param penvalueridge_list:
    :param penfirstridge_list:
    :param pensecondridge_list:
    :param learning_rate_list:
    :param patience_list:
    :param min_delta_list:
    :param degree_transformations_list:
    :param degree_decorrelation_list:
    :return:
    """

    list_of_lists = [penvalueridge_list, penfirstridge_list, pensecondridge_list,
                     lambda_penalty_params_list,
                     learning_rate_list,
                     patience_list, min_delta_list,
                     degree_transformations_list, degree_decorrelation_list]
                     #normalisation_layer_list]
    #TODO: also allow for grid search, required for the lasso
    hyperparameter_combinations_list = list(itertools.product(*list_of_lists))

    if tuning_mode == "optuna":
        penvalueridge, penfirstridge, pensecondridge, lambda_penalty_params, learning_rate, \
        patience, min_delta, degree_transformations, degree_decorrelation  = hyperparameter_combinations_list[0]

        def optuna_objective(trial, y_train=y_train, y_validate=y_validate):

            # Defining the model
            poly_range = torch.FloatTensor([[-poly_span_abs], [poly_span_abs]])

            #penvalueridge_opt  = trial.suggest_float("penvalueridge", 0.001, 5, log=True),
            penvalueridge_opt = 0
            penfirstridge_opt  = trial.suggest_float("penfirstridge", 0.001, 10, log=True)
            pensecondridge_opt = trial.suggest_float("pensecondridge", 0.001, 20, log=True)
            penalty_params = torch.tensor([penvalueridge_opt,
                                           penfirstridge_opt,
                                           pensecondridge_opt])

            if tune_precision_matrix_penalty == True:
                lambda_penalty_params_opt = trial.suggest_float("lambda_penalty_params", 0.000001, 1, log=True)
                num_vars = y_train.size()[1]
                lambda_penalty_params_opt = torch.tensor(lambda_penalty_params_opt).repeat(num_vars,num_vars) - torch.eye(num_vars,num_vars) * lambda_penalty_params_opt
            else:
                lambda_penalty_params_opt = lambda_penalty_params

            # for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(y.size()[0]))):
            #    y_train = y[train_idx, :]
            #    y_validate = y[val_idx, :]

            if x_train is False:
                number_covariates = 0
            else:
                number_covariates = 1

            #Cross validation inspired by:
            #https://stackoverflow.com/questions/63224426/how-can-i-cross-validate-by-pytorch-and-optuna

            if cross_validation_folds == False:

                y_train = y_train.to(device)
                y_validate = y_validate.to(device)

                nf_mctm = NF_MCTM(input_min=y_train.min(0).values,
                                  input_max=y_train.max(0).values,
                                  polynomial_range=poly_range,
                                  number_variables=y_train.size()[1],
                                  spline_decorrelation=spline_decorrelation,
                                  degree_transformations=degree_transformations,
                                  degree_decorrelation=degree_decorrelation,
                                  number_covariates=number_covariates,
                                  device=device)
                # normalisation_layer=normalisation_layer)

                nf_mctm = nf_mctm.to(device)

                train(model=nf_mctm,
                      train_data=y_train,
                      validate_data=y_validate,
                      train_covariates=x_train,
                      validate_covariates=x_validate,
                      penalty_params=penalty_params,
                      lambda_penalty_params=lambda_penalty_params_opt,
                      iterations=iterations,
                      learning_rate=learning_rate,
                      patience=patience,
                      min_delta=min_delta,
                      verbose=False,
                      return_report=False)  # no need for reporting and metrics,plots etc.

                return nf_mctm.log_likelihood(y_validate, x_validate).cpu().detach().numpy().sum()

            else:
                kf = KFold(n_splits=cross_validation_folds, shuffle=True)
                log_likelihoods = []
                for train_idx, val_idx in kf.split(np.arange(y_train.size()[0])):
                    y_train_cv = y_train[train_idx, :]
                    y_validate_cv = y_train[val_idx, :]

                    if x_train is False:
                        x_train_cv = False
                        x_validate_cv = False
                    else:
                        x_train_cv = x_train[train_idx, :]
                        x_validate_cv = x_train[val_idx, :]

                    y_train_cv = y_train_cv.to(device)
                    y_validate_cv = y_validate_cv.to(device)

                    nf_mctm = NF_MCTM(input_min=y_train_cv.min(0).values,
                                      input_max=y_train_cv.max(0).values,
                                      polynomial_range=poly_range,
                                      number_variables=y_train_cv.size()[1],
                                      spline_decorrelation=spline_decorrelation,
                                      degree_transformations=degree_transformations,
                                      degree_decorrelation=degree_decorrelation,
                                      number_covariates=number_covariates,
                                      device=device)
                    # normalisation_layer=normalisation_layer)

                    nf_mctm = nf_mctm.to(device)

                    #print("nf_mctm",nf_mctm.device)
                    #print("y_train_cv", y_train_cv.device)
                    #print("y_validate_cv", y_validate_cv.device)

                    train(model=nf_mctm,
                          train_data=y_train_cv,
                          validate_data=y_validate_cv,
                          train_covariates=x_train_cv,
                          validate_covariates=x_validate_cv,
                          penalty_params=penalty_params.to(device),
                          lambda_penalty_params=lambda_penalty_params_opt,
                          iterations=iterations,
                          learning_rate=learning_rate,
                          patience=patience,
                          min_delta=min_delta,
                          verbose=False,
                          return_report=False)

                    log_likelihoods.append(nf_mctm.log_likelihood(y_validate_cv, x_validate_cv).cpu().detach().numpy().sum())
                return np.mean(log_likelihoods)

        # docs: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
        study = optuna.create_study(sampler=TPESampler(n_startup_trials=7,
                                                       consider_prior=True, # is this useful without a prior weight?
                                                       prior_weight=0, #default value 1.0 but then does not explore the space as good I think
                                                       multivariate=True # experimental but very useful here as our parameters are highly correlated
                                                       ),
                                    direction='maximize')
        study.optimize(optuna_objective, n_trials=15)

        print("hyperparameter_tuning done")

    #elif tuning_mode == "grid":
    #list_of_lists = [penvalueridge_list, penfirstridge_list, pensecondridge_list,
    #                 learning_rate_list,
    #                 patience_list, min_delta_list,
    #                 degree_transformations_list, degree_decorrelation_list]
    #                 #normalisation_layer_list]
    #hyperparameter_combinations_list = list(itertools.product(*list_of_lists))

    #splits = KFold(n_splits=3) #want larger percent of data in train set? maybe another sampling method?
                          # parallelisation even of the folds? 

    #results = pd.DataFrame(columns=['penvalueridge', 'penfirstridge', 'pensecondridge', 'learning_rate',
    #                                'patience', 'min_delta', 'degree_transformations',
    #                                'degree_decorrelation', #'normalisation_layer',
    #                                #'fold',
    #                                'sum_validation_log_likelihood'])
#
    #for hyperparamters in hyperparameter_combinations_list:
    #    penvalueridge, penfirstridge, pensecondridge, learning_rate, \
    #    patience, min_delta, degree_transformations, degree_decorrelation  = hyperparamters #normalisation_layer
#
    #    # Defining the model
    #    poly_range = torch.FloatTensor([[-poly_span_abs], [poly_span_abs]])
    #    penalty_params = torch.tensor([penvalueridge,
    #                                   penfirstridge,
    #                                   pensecondridge])
#
    #    #for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(y.size()[0]))):
    #    #    y_train = y[train_idx, :]
    #    #    y_validate = y[val_idx, :]
#
    #    nf_mctm = NF_MCTM(input_min=y_train.min(0).values,
    #                      input_max=y_train.max(0).values,
    #                      polynomial_range=poly_range,
    #                      number_variables=y_train.size()[1],
    #                      spline_decorrelation=spline_decorrelation,
    #                      degree_transformations=degree_transformations,
    #                      degree_decorrelation=degree_decorrelation)
    #                      #normalisation_layer=normalisation_layer)
#
    #    train(model=nf_mctm,
    #          train_data=y_train,
    #          penalty_params=penalty_params,
    #          iterations=iterations,
    #          learning_rate=learning_rate,
    #          patience=patience,
    #          min_delta=min_delta,
    #          verbose=False,
    #          return_report=False) #no need for reporting and metrics,plots etc...
#
    #    sum_validation_log_likelihood = nf_mctm.log_likelihood(y_validate).detach().numpy().sum()

        #results = results.append({'penvalueridge': penvalueridge, 'penfirstridge': penfirstridge,
        #                          'pensecondridge': pensecondridge, 'learning_rate': learning_rate,
        #                          'patience': patience, 'min_delta': min_delta,
        #                          'degree_transformations': degree_transformations,
        #                          'degree_decorrelation': degree_decorrelation,
        #                          #'normalisation_layer': normalisation_layer,
        #                          #'fold': fold,
        #                          'sum_validation_log_likelihood': sum_validation_log_likelihood},
        #                         ignore_index=True)
#
    return study


def extract_optimal_hyperparameters(results: pd.DataFrame):
    """
    Extracts the optimal hyperparameters from the results dataframe
    :param results:
    :return:
    """

    results_mean = results.groupby(['penvalueridge', 'penfirstridge', 'pensecondridge', 'learning_rate',
                                    'patience', 'min_delta', 'degree_transformations',
                                    'degree_decorrelation']).mean()
    results_std = results.groupby(['penvalueridge', 'penfirstridge', 'pensecondridge', 'learning_rate',
                                    'patience', 'min_delta', 'degree_transformations',
                                    'degree_decorrelation']).std()
    #TODO: cannot grooubby if we have nan values in a column, e.g. dropped normalisation_layer, if I decide to keep this layer then fix this
    results_mean = results_mean.reset_index()
    results_std = results_std.reset_index()

    optimal_hyperparameters = results_mean.loc[results_mean['sum_validation_log_likelihood'].idxmax()]
    if 'fold' in results_mean.columns:
        optimal_hyperparameters = optimal_hyperparameters.drop(['fold'])
    optimal_hyperparameters = optimal_hyperparameters.drop(['sum_validation_log_likelihood'])

    results_mean["std_validation_log_likelihood"] = results_std["sum_validation_log_likelihood"]
    results_mean.rename(columns={'sum_validation_log_likelihood': 'mean_validation_log_likelihood)'}, inplace=True)
    if 'fold' in results_mean.columns:
        results_moments = results_mean.drop(['fold'],axis=1)
    else:
        results_moments = results_mean
    return optimal_hyperparameters.values, results_moments

