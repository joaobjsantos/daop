import dataset.data_processing_no_ssl as data_processing_no_ssl
import no_ssl

import configs.config_base_quick_test as config_base

ROTNET_DA = [[0, [1.0, 0.2, 0.2, 0.2, 0.2]], [1, [0.5, 0.5, 0.5, 0.5, 0.5]]]

config = config_base.config
config['fix_pretext_da'] = []
config['base_experiment_name'] = "optimize_po"
config['experiment_name'] = config['base_experiment_name']
config['output_csv_folder'] = "output_csv" + "_" + config['base_experiment_name']
config['fix_downstream_da'] = ROTNET_DA  
