from eICU_preprocessing.split_train_test import create_folder
from models.run_lstm import BaselineLSTM
from models.initialise_arguments import initialise_lstm_arguments
from models.final_experiment_scripts.best_hyperparameters import best_lstm


if __name__=='__main__':

    c = initialise_lstm_arguments()
    c['exp_name'] = 'StandardLSTM50_no_labs_no_diagnosis'
    c['dataset'] = 'eICU'
    c['percentage_data'] = 50
    c['no_labs'] = True
    c['no_diag'] = True
    c = best_lstm(c)

    log_folder_path = create_folder('models/experiments/final/eICU/LoS', c.exp_name)
    baseline_lstm = BaselineLSTM(config=c,
                                 n_epochs=c.n_epochs,
                                 name=c.exp_name,
                                 base_dir=log_folder_path,
                                 explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    baseline_lstm.run()