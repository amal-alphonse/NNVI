import optuna

from nnvi.config import Config
from nnvi.train import train_cfg

# Define objective function for Optuna to optimise
def objective(trial):
    # Define the hyperparameter search spaces 
    o_lr_soln = trial.suggest_loguniform('learning_rate_1', 1e-5, 1e-1)
    o_lr_testfn = trial.suggest_loguniform('learning_rate_2', 1e-5, 1e-1)

    o_weight_soln_obs = trial.suggest_categorical("weight_soln_obs",   [1, 10, 100, 1000, 10000])
    o_weight_testfn_obs = trial.suggest_categorical("weight_testfn_obs",   [1, 10, 100, 1000, 10000])
    o_gamma_soln = trial.suggest_categorical("gamma_1", [0.2, 0.4, 0.6, 0.8])
    o_gamma_testn = trial.suggest_categorical("gamma_2", [0.2, 0.4, 0.6, 0.8])
    o_diagonal = trial.suggest_categorical("weight_gap_term", [0.0001, 0.0005, 0.0002, 0.001])
  
    # o_lr_soln = trial.suggest_loguniform('learning_rate_1', 1e-5, 1e-1)
    # o_lr_testfn = trial.suggest_loguniform('learning_rate_2', 1e-5, 1e-1)

    # o_weight_b = trial.suggest_int("weight_b", 1e2, 1e6)
    # o_weight_soln_obs = trial.suggest_int("weight_soln_obs",  1e2, 1e6)
    # o_weight_testfn_obs = trial.suggest_int("weight_testfn_obs",  1e2, 1e6)
    # o_weight_b_test = trial.suggest_int("weight_b_test", 1e2, 1e6)
    # o_gamma_soln = trial.suggest_categorical("gamma_1", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # o_gamma_testn = trial.suggest_categorical("gamma_2", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # o_diagonal = trial.suggest_loguniform("weight_gap_term", 1e-5, 1e-1)

    conf = Config(example="NSV_two_dim", 
    n_interior = 1024, 
    n_boundary = 256,  
    epochs = 10000, 
    soln_model_name = "DrrnnWithSpecificBC", 
    soln_model_args =  {"width": 80, "depth": 4, "activation":"Tanh", "hc_bias": 0.0},

    lr_soln  = o_lr_soln,
    lr_testfn = o_lr_testfn,
    lr_scheduler_soln_args = {"gamma": o_gamma_soln, "step_size": 1000},
    lr_scheduler_testfn_args  = {"gamma": o_gamma_testn, "step_size": 1000},
    seed = 123,
    optim_soln = "AdamW",
    optim_testfn  = "AdamW",
    weight_b = 1,
    weight_soln_obs = o_weight_soln_obs,
    weight_testfn_obs = o_weight_testfn_obs,
    weight_b_test = 1,
    print_freq = 100,
    eval_freq = 5,
    weight_gap_term  = o_diagonal,
    testfn_model_name = "DrrnnWithSpecificBC", 
    testfn_model_args =  {"width": 80, "depth": 4, "activation":"Tanh", "hc_bias": 0.0},
    loss_name = "RegularisedGapLoss",
    use_H1_norm_for_gap_term= True,
    force_bc= False,
    chpt_folder= './plots_optuna/',
)

    print("Starting config: ", conf)
    error = train_cfg(conf)
    
    # Return the performance metric to be optimised
    return error

if __name__ == "__main__":
    # Set your input and output dimensions

    def print_best_callback(study, trial):
        print(f"\n Best value: {study.best_value},\n Best params: {study.best_trial.params}\n")

    # Create a study object and optimise the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, callbacks=[print_best_callback])  # n_trials is the number of optimisation iterations

    # Print the best hyperparameters and corresponding value of the objective function
    print("Best hyperparameters: ", study.best_params)
    print("Best objective value: ", study.best_value)