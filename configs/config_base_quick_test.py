import numpy as np
import torch

import os

import dataset.data_processing_rotnet_cifar as data_processing_rotnet_cifar
import DA.data_augmentation_albumentations as data_augmentation_albumentations
import net_models_torch
import rotnet_torch
import state_manager_torch
import mutations
import chromosomes
import train_with_DA
import evolution_mod_functions

ROTNET_DA = [[0, [1.0, 0.2, 0.2, 0.2, 0.2]], [1, [0.5, 0.5, 0.5, 0.5, 0.5]]]

import configs.config_base as config_base

config = config_base.config

# model training configs
config['base_epochs'] = 2
config['epochs'] = config['base_epochs']

config['start_gen'] = 1
config['stop_gen'] = 10
config['population_size'] = 2
config['max_chromosomes'] = 2
config['evolution_mods'] = {
    # 201: evolution_mod_functions.extended_gens
}