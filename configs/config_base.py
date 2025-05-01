import numpy as np
import torch

import os

import data_processing_rotnet_torch
import data_augmentation_torch
import net_models_torch
import rotnet_torch
import state_manager_torch
import mutations
import chromosomes
import train_with_DA
import evolution_mod_functions

ROTNET_DA = [[0, [1.0, 0.2, 0.2, 0.2, 0.2]], [1, [0.5, 0.5, 0.5, 0.5, 0.5]]]
# FIX_PRETEXT_DA = [[0, [1.0, 0.2, 0.2, 0.2, 0.2]], [1, [1.0, 0.5, 0.5, 0.5, 0.5]]]
# FIX_DOWNSTREAM_DA = [[0, [1.0, 0.2, 0.2, 0.2, 0.2]], [1, [0.5, 0.5, 0.5, 0.5, 0.5]]]

# START_PARENT = [[43, [0.33, 0.56, 0.35, 0.32]]]

config = {}

# experiment configs
config['base_experiment_name'] = "optimize_simultaneous_100"
config['experiment_name'] = config['base_experiment_name']
config['seeds'] = range(5)
config['seed'] = config['seeds'][0]
config['state_folder'] = "states"
config['state_file'] = None
config['load_state'] = state_manager_torch.load_state
config['save_state'] = state_manager_torch.save_state
config['every_gen_state_reset'] = None

# dataset configs
config['dataset'] = "cifar10"
config['dataset_file_name'] = "cifar-10-python.tar.gz"
config['dataset_transforms'] = data_processing_rotnet_torch.dataset_transforms
config['dim'] = (32, 32, 3)
config['num_classes'] = 10
config['num_classes_pretext'] = 4
config['num_classes_downstream'] = config['num_classes']
config['load_dataset_func'] = data_processing_rotnet_torch.load_dataset
config['data_loader_func'] = data_processing_rotnet_torch.create_data_loaders
config['cache_folder'] = "cache_cifar10_torch"
config['output_csv_folder'] = "output_csv" + "_" + config['base_experiment_name']
config['delete_cache'] = False

# augmentation configs
config['individual_evaluation_func'] = train_with_DA.train_and_evaluate_individual
config['augment_dataset'] = True
config['min_da_prob'] = 0.1
config['max_da_prob'] = 0.9
config['da_funcs'] = data_augmentation_torch.da_funcs_probs(config['min_da_prob'], config['max_da_prob'], config['dim'][:2])
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {config['device']}")
# config['shuffle_dataset'] = False     # shuffle with sampler
config['evolution_type'] = "simultaneous"   # "same"/"simultaneous"
config['fix_pretext_da'] = None     # FIX_PRETEXT_DA
config['fix_downstream_da'] = None      # FIX_DOWNSTREAM_DA
# config['base_pretrained_pretext_model'] = os.path.join("models", "rn_pretext_0.pt")
# config['pretrained_pretext_model'] = config['base_pretrained_pretext_model']
# config['extended_pretrained_pretext_model'] = os.path.join("models", "rn_100_pretext_6.pt")
config['base_pretrained_pretext_model'] = None
config['pretrained_pretext_model'] = None
config['extended_pretrained_pretext_model'] = None

# model training configs
config['framework'] = 'torch'
config['isSSL'] = True
config['model'] = net_models_torch.TrainResNet18
config['finetune_backbone'] = False
config['base_epochs'] = 100
config['epochs'] = config['base_epochs']
config['extended_epochs'] = 100
config['base_pretext_epochs'] = lambda: config['epochs']
config['base_downstream_epochs'] = lambda: config['epochs']
config['pretext_epochs'] = config['base_pretext_epochs']
config['downstream_epochs'] = config['base_downstream_epochs']
config['extended_pretext_epochs'] = lambda: config['extended_epochs']
config['extended_downstream_epochs'] = lambda: config['extended_epochs']
config['batch_size'] = 128
config['pretext_batch_size'] = lambda: config['batch_size'] * 4
config['downstream_batch_size'] = lambda: config['batch_size']
config['num_workers'] = 4
config['model_evaluate_func'] = rotnet_torch.evaluate_rotnet
config['check_memory_leaks'] = False
config['track_train_bottleneck'] = False
config['save_models_folder'] = None
config['save_pretext_model'] = None
config['save_downstream_model'] = None
# config['save_models_folder'] = "models"
# config['save_pretext_model'] = config['base_experiment_name'] + "_pretext"
# config['save_downstream_model'] = config['base_experiment_name'] + "_downstream"
config['shuffle_dataset'] = True
config['confusion_matrix_config'] = None
# config['confusion_matrix_config'] = {
#     'print_confusion_matrix': True,
#     'confusion_matrix_folder': "output_confusion_matrix",
#     'confusion_matrix_pretext_file': "confusion_matrix_pretext.txt",
#     'confusion_matrix_downstream_file': "confusion_matrix_downstream.txt",
#     'num_classes_pretext': config['num_classes_pretext'],
#     'num_classes_downstream': config['num_classes_downstream']
# }
# config['torch_float_type'] = torch.float16

# evolutionary algorithm configs
config['n_pr'] = 4
config['start_parent'] = None
config['start_population'] = None
config['best_n'] = 5
config['best_individuals'] = None
config['extended_isolated_run'] = False
config['current_run_generations'] = 0
config['max_generations_per_run'] = None
config['start_gen'] = 1
config['stop_gen'] = 200
config['population_size'] = 5
config['max_chromosomes'] = 5
config['recalculate_best'] = True
config['create_da_func'] = chromosomes.random_da_func(len(config['da_funcs']))
config['create_pr'] = chromosomes.random_pr
config['create_chromosome'] = chromosomes.create_chromosome_2_levels(config['create_da_func'], config['create_pr'], config['n_pr'])
config['da_func_mutation'] = chromosomes.random_da_func(len(config['da_funcs']))
config['pr_mutation'] = chromosomes.random_pr_gaussian(0.1)
config['mutation'] = mutations.mutate_remove_change_add_seq(0.66, 0.33, 0.66)
config['evolution_mods'] = {
    # 201: evolution_mod_functions.extended_gens
}