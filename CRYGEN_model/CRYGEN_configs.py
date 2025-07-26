import torch


ALL_CONFIGS = {
    "default": {
        "training_config": {
            "optimizer": {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.0,
            },
            "use_lr_scheduler": True,
            "lr_scheduler": {
                "factor": 0.6,
                "patience": 30,
                "min_lr": 1e-4,
            },
            "deterministic": True,
            "random_seed": 42,
            "epochs": 1000,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "early_stopping_patience": 100000,
            "save_path": "CRYGENdemo",
            "guidance_path": "CRYGENdemo-guidance",
        },
        "datamodule_config": {
            "dataset_configs": {
                "train": {
                    "name": "Formation energy train",
                    "path": "MP20_dataset/train.csv",
                    "save_path": "MP20_dataset/train_ori.pt",
                    "prop_types": {"meta_stable": "binary"},
                    "niggli": True,
                    "primitive": False,
                    "graph_method": "crystalnn",
                    "tolerance": 0.1,
                    "use_space_group": False,
                    "use_pos_index": False,
                    "lattice_scale_method": "scale_length",
                    "preprocess_workers": 30,
                },
                "val": {
                    "name": "Formation energy val",
                    "path": "MP20_dataset/val.csv",
                    "save_path": "MP20_dataset/val_ori.pt",
                    "prop_types": {"meta_stable": "binary"},
                    "niggli": True,
                    "primitive": False,
                    "graph_method": "crystalnn",
                    "tolerance": 0.1,
                    "use_space_group": False,
                    "use_pos_index": False,
                    "lattice_scale_method": "scale_length",
                    "preprocess_workers": 30,
                },
                "test": [{
                    "name": "Formation energy test",
                    "path": "MP20_dataset/test.csv",
                    "save_path": "MP20_dataset/test_ori.pt",
                    "prop_types": {"meta_stable": "binary"},
                    "niggli": True,
                    "primitive": False,
                    "graph_method": "crystalnn",
                    "tolerance": 0.1,
                    "use_space_group": False,
                    "use_pos_index": False,
                    "lattice_scale_method": "scale_length",
                    "preprocess_workers": 30,
                }]
            },
            "num_workers": {
                "train": 0,
                "val": 0,
                "test": 0,
            },
            "batch_size": {
                "train": 256,
                "val": 128,
                "test": 128,
            }
        }
    },
    "e_hull": {
        "training_config": {
            "pretrain_path": "CRYGENdemo_best.pt",
            "save_path": "CRYGENdemo_e_hull",
            "optimizer": {
                "lr": 0.0005
            },
            "inference_using": "best"
        },
        "datamodule_config": {
            "dataset_configs": {
                "train": {
                    "path": "MP20_dataset/train_w_comp_ehull.csv",
                    "save_path": "MP20_dataset/train_ehull_ori.pt",
                    "prop_types": {"meta_stable": "binary", "mattersim_ehull": "continuous"},
                    "prop_special_values": {"mattersim_ehull": [-1.1, 9.9]}
                },
                "val": {
                    "path": "MP20_dataset/val_w_comp_ehull.csv",
                    "save_path": "MP20_dataset/val_ehull_ori.pt",
                    "prop_types": {"meta_stable": "binary", "mattersim_ehull": "continuous"},
                    "prop_special_values": {"mattersim_ehull": [-1.1, 9.9]}
                }
            }
        }
    },
    "phonon_stability": {
        "training_config": {
            "pretrain_path": "CRYGENdemo_best.pt",
            "save_path": "CRYGENdemo_phonon_stability",
            "optimizer": {
                "lr": 0.0005
            },
            "inference_using": "best"
        },
        "datamodule_config": {
            "dataset_configs": {
                "train": {
                    "path": "MP20_dataset/train_phonon_data.csv",
                    "save_path": "MP20_dataset/train_phonon_data_ori.pt",
                    "prop_types": {"meta_stable": "binary", "mattersim_phonon_stability": "binary"},
                },
                "val": {
                    "path": "MP20_dataset/val_phonon_data.csv",
                    "save_path": "MP20_dataset/val_phonon_data_ori.pt",
                    "prop_types": {"meta_stable": "binary", "mattersim_phonon_stability": "binary"},
                }
            }
        }
    },
    "phonon_stability_kappa": {
        "training_config": {
            "pretrain_path": "CRYGENdemo_phonon_stability_best.pt",
            "save_path": "CRYGENdemo_phonon_stability_kappa",
            "optimizer": {
                "lr": 0.0001
            },
            "inference_using": "best"
        },
        "datamodule_config": {
            "dataset_configs": {
                "train": {
                    "path": "MP20_dataset/train_mini_kappa.csv",
                    "save_path": "MP20_dataset/train_mini_kappa_ori.pt",
                    "prop_types": {"meta_stable": "binary", "mattersim_phonon_stability": "binary", "scaled_kappa": "continuous"},
                },
                "val": {
                    "path": "MP20_dataset/val_mini_kappa.csv",
                    "save_path": "MP20_dataset/val_mini_kappa_ori.pt",
                    "prop_types": {"meta_stable": "binary", "mattersim_phonon_stability": "binary", "scaled_kappa": "continuous"},
                }
            }
        }
    }
}


def update_nested_dict(config1, config2):
    for key, value in config2.items():
        if isinstance(value, dict):
            if key in config1:
                config1[key] = update_nested_dict(config1[key], value)
            else:
                config1[key] = value
        else:
            config1[key] = value
    return config1


def get_config(config_name: str = ""):
    default_config = ALL_CONFIGS["default"]
    if not config_name:
        return default_config
    elif config_name in ALL_CONFIGS:
        return update_nested_dict(default_config, ALL_CONFIGS[config_name])
    else:
        raise ValueError(f"Config name {config_name} not found")
