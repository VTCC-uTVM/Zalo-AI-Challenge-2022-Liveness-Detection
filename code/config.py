import optuna

def suggest_hyperparameters(trial):
    # Learning rate on a logarithmic scale
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # Dropout ratio in the range from 0.0 to 0.9 with step size 0.1
    dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
    # Optimizer to use as categorical value
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta"])

    return lr, dropout, optimizer_name