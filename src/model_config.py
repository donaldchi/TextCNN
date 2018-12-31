DEFAULT_CONFIG = {
    "embedding_dim": 200,
    "filter_sizes": "3,4,5",
    "num_filters": 128,
    "dropout_keep_prob": 0.5,
    "l2_reg_lambda": 0.0,
    "batch_size": 1000,
    "num_epochs": 10,
    "evaluate_every": 10,
    "checkpoint_every": 100,
    "num_checkpoints": 5,
    'use_k_fold': False,
    'use_multi_channel': False,
    'use_pretrained_embedding': False,
    "allow_soft_placement": True,  # Allow device soft device placement
    "log_device_placement": False  # Log placement of ops on devices
}
