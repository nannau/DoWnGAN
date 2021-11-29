from mlflow import log_param
import hyperparams as hp
import logging

def log_hyperparams():
    """Logs the hyperparameters"""
    keys = [item for item in dir(hp) if not item.startswith("__")]
    values = hp.__dict__
    for key in values:
        log_param(key, values[key])


def define_experiment(mlclient):
    print("Enter the experiment name you wish to add the preceding training run to.")
    print("Select number from list or press n for new experiment: ")
    [print(exp.experiment_id,":", exp.name) for i, exp in enumerate(mlclient.list_experiments())]
    again = False
    while again is False:
        choice = input("Input number here: ")
        if choice == "n":
            g = True
            while g is True:
                set_exp = input("Enter new descriptive experiment name ")
                confirm = input(f"You entered {set_exp}. Happy? (Y/n) ")
                if confirm == "Y" or confirm == '':
                    mlclient.create_experiment(set_exp)
                    g = False
            again = True

        elif choice.isnumeric():
            g = True
            set_exp = mlclient.get_experiment(choice).name
            while g is True:
                confirm = input(f"You have selected {set_exp}. Happy? (Y/n) ")
                if confirm == "Y" or confirm == '':
                    g = False
            again = True

        else:
            print("Please select a valid input")

    return mlclient.get_experiment_by_name(set_exp).experiment_id

def write_tags():
    g = True
    while g is True:
        choice = input("Describe the specifics and purpose of this training run: ")
        confirm = input(f"You entered {choice}. Happy? (Y/n) ")
        if confirm == "Y" or confirm == '':
            g = False
    return choice