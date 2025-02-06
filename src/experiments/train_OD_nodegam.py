import os
import pickle
import time
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import csv
import nodegam
from feature_utils import process

parameters = {
    'format':'OD',
    'anneal_steps' : 5000,
    'quantile_noise':1e-4,
    'n_quantiles':3000,
    'min_temp':0.1,
    'num_trees':300,
    'num_layers':8,
    'depth':6,
    'lr':1e-4,
    'lr_warmup_steps':1000,
    'batch_size':128,
    'lr_decay_steps' : 5000,
    'early_stopping_rounds' : 1000,
    'output_dropout':0.2,
    'last_dropout':0.3,
    'colsample_bytree':0.5,
    'selectors_detach':0,
    'ga2m':1,
    'l2_lambda':0.3,
    'nus_min':0.7,
    'nus_max':1.0,
}

# Only use GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
name = 'OD_ga2m_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
# Create directory
os.makedirs(pjoin('logs', name), exist_ok=True)

csv_path = "./logs/parameters.csv"
if os.path.exists(csv_path):
    parameters_df = pd.read_csv(csv_path)
    columns_to_check = [col for col in parameters_df.columns if col in parameters]
    matching_rows = parameters_df[columns_to_check].eq(pd.Series(parameters)).all(axis=1)
    if matching_rows.any():
        print(f"Already trained a model on this parameter with name : {parameters_df['name']} and test_error : {parameters_df['test_error']}")
        exit(0)

# Set seed
nodegam.utils.seed_everything(seed=83)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

format = "OD"
train_data_path = os.path.join("..", "data", "processed", "train", f"15_{format}.csv")
test_data_path = os.path.join("..", "data", "processed", "test", f"15_{format}.csv")

test_df = pd.read_csv(test_data_path)
train_df = pd.read_csv(train_data_path)

y_train = train_df["fantasy_points"].values.squeeze()
y_test_id = test_df["match_id"]
y_test = test_df["fantasy_points"].values.squeeze()


X_train = process(train_df, 15, False)
X_test = process(test_df, 15, False)

data = {
    'X_train' : X_train,
    'y_train' : y_train,
    'X_test' : X_test,
    'y_test' : y_test,
    'problem' : "regression"
}

preprocessor = nodegam.mypreprocessor.MyPreprocessor(
    cat_features=data.get('cat_features', None),
    y_normalize=(data['problem'] == 'regression'), # Normalize target y to mean 0 and 1 in regression
    random_state=1337, quantile_transform=True,
    quantile_noise=data.get('quantile_noise', parameters['quantile_noise']),
    n_quantiles=parameters['n_quantiles'],
)

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
preprocessor.fit(X_train, y_train)

X_train, y_train = preprocessor.transform(X_train, y_train)
X_test, y_test = preprocessor.transform(X_test, y_test)

anneal_steps = parameters['anneal_steps']

choice_fn = nodegam.nn_utils.EM15Temp(max_temp=1., min_temp=parameters['min_temp'], steps=anneal_steps)

# Temperature annealing for entmoid
model = nodegam.arch.GAMBlock(
    in_features=X_train.shape[1],
    num_trees=parameters['num_trees'],
    num_layers=parameters['num_layers'],
    num_classes=1,
    addi_tree_dim=1,
    depth=parameters['depth'],
    choice_function=choice_fn,
    bin_function=nodegam.nn_utils.entmoid15,
    output_dropout=parameters['output_dropout'],
    last_dropout=parameters['last_dropout'],
    colsample_bytree=parameters['colsample_bytree'],
    selectors_detach=parameters['selectors_detach'], # This is only used to save memory in large datasets like epsilon
    add_last_linear=True,
    ga2m=parameters['ga2m'],
    l2_lambda=parameters['l2_lambda'],
).to(device)

step_callbacks = [choice_fn.temp_step_callback]

from qhoptim.pyt import QHAdam
optimizer_params = {'nus': (parameters['nus_min'], parameters['nus_max']), 'betas': (0.95, 0.998)}

trainer = nodegam.trainer.Trainer(
    model=model,
    experiment_name=name,
    warm_start=True, # if True, will load latest checkpt in the saved dir logs/${name}
    Optimizer=QHAdam,
    optimizer_params=optimizer_params,
    lr=parameters['lr'],
    lr_warmup_steps=parameters['lr_warmup_steps'],
    verbose=False,
    n_last_checkpoints=5,
    step_callbacks=step_callbacks, # Temp annelaing
    fp16=1,
    problem=data['problem'],
)
batch_size = parameters['batch_size']

loss_history, err_history, err_UMSE_history = [], [], []
report_frequency = 100
best_err, best_step_err = np.inf, -1
early_stopping_rounds = parameters['early_stopping_rounds']
lr_decay_steps = parameters['lr_decay_steps']
prev_lr_decay_step = 0
max_rounds = -1 # No max round set
max_time = 3600 * 10 # 10 hours
best_step_err = 0
best_err = float('inf')
patience_counter = 0
st_time = time.time()

for batch in nodegam.utils.iterate_minibatches(X_train, y_train,
                                               batch_size=batch_size,
                                               shuffle=True, epochs=float('inf')):
    metrics = trainer.train_on_batch(*batch, device=device)
    loss_history.append(float(metrics['loss']))

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.remove_old_temp_checkpoints()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')

        err = trainer.evaluate_mse(X_test, y_test, device=device, batch_size=batch_size * 2)
        
        # Update best test error
        if err < best_err:
            best_err = err
            best_step_err = trainer.step
            patience_counter = 0  # Reset patience counter
            trainer.save_checkpoint(tag='best')
        else:
            patience_counter += 1  # Increment counter for no improvement
        
        err_history.append(err)
        err_UMSE_history.append(err* (preprocessor.y_std) ** 2)

        trainer.load_checkpoint()  # Load last state

        # Plot loss and error curves
        clear_output(True)
        plt.figure(figsize=[18, 6])
        plt.subplot(1, 3, 1)
        plt.plot(loss_history)
        plt.title('Loss')
        plt.grid()
        plt.subplot(1, 3, 2)
        plt.plot(err_history)
        plt.title('Error')
        plt.grid()
        plt.subplot(1, 3, 3)
        plt.plot(err_UMSE_history)
        plt.title('Error_UMSE')
        plt.grid()
        plt.show()
        plt.savefig(f'./logs/{name}/loss.png')

    # Early stopping condition
    if patience_counter > early_stopping_rounds:
        print('BREAK. No improvement for {} steps.'.format(early_stopping_rounds))
        break

    if trainer.step == 8500 :
        break

    # Learning rate decay
    if lr_decay_steps > 0 \
            and trainer.step > best_step_err + lr_decay_steps \
            and trainer.step > (prev_lr_decay_step + lr_decay_steps):
        lr_before = trainer.lr
        trainer.decrease_lr(ratio=0.2, min_lr=1e-6)
        prev_lr_decay_step = trainer.step
        print('LR: %.2e -> %.2e' % (lr_before, trainer.lr))

    # Hard stops
    if 0 < max_rounds < trainer.step:
        print('End. Maximum rounds reached: %d' % max_rounds)
        break

    if (time.time() - st_time) > max_time:
        print('End. Maximum runtime reached: %d seconds' % max_time)
        break

print("Best step: ", best_step_err)
print("Best test Error: ", best_err)
print("unnormliased MSE: ", best_err * (preprocessor.y_std) ** 2)

parameters['name'] = name
parameters['test_error'] = best_err
parameters['unnormlaised MSE'] = best_err * (preprocessor.y_std) ** 2

fieldnames = list(parameters.keys())

with open(csv_path, mode="a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    if file.tell() == 0:
        writer.writeheader()
    
    writer.writerow(parameters)  
