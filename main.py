import random
from run_modele import model_to_test
import torch
from tqdm import tqdm
import json
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

search_param = {
    "nb_epoch": 2000,  # epoch number
    "save_rate": 50,  # rate to save
    "weight_data": 1,
    "weight_pde": 1,
    "nb_points_pde": 1000000,  # Total number of pde points
    "Re": 100,
    "gamma_scheduler": 0.999,  # Gamma scheduler for lr
    "nb_layers": 1,
    "nb_neurons": 32,
    "n_pde_test": 5000,
    "n_data_test": 5000,
    "x_min": 0.15,
    "x_max": 0.325,
    "y_min": -0.1,
    "y_max": 0.1,
    "t_min": 4,
    "t_max": 5,
    "transfert_learning": "None",
}

lr_min = 1e-4
lr_max = 1e-3
list_batch_size = [16, 32, 64, 128]
nb_axes_min = 6
nb_axes_max = 30
save_folder = 'piche'
time_run = 10


int_parameter = {
    "lr_min":lr_min,
    "list_batch_size":list_batch_size,
    "nb_axes_min":nb_axes_min,
    "nb_axes_max":nb_axes_max,
    "time_run":time_run
}

Path('results/' + save_folder).mkdir(parents=True, exist_ok=True)  # Creation du dossier de result
with open('results/' + save_folder + "/param_simu.json", "w") as file:
    json.dump(search_param, file, indent=4)
    json.dump(int_parameter, file, indent=4)



for num_sim in tqdm(range(3)):
    print('\n\n\n-------------------------')
    print(f"Simu n°{num_sim+1}")
    print('-------------------------\n')
    lr_init = random.uniform(1e-3, 1e-4)
    batch_size = random.choice(list_batch_size)
    nb_axes = random.randint(nb_axes_min, nb_axes_max)
    to_test = model_to_test(
        batch_size,
        nb_axes,
        lr_init,
        save_folder,
        time_run,
        num_sim,
        device,
        search_param
    )
    to_test.run()
    
    