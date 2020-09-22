import os


class Arg:
    random_seed: int = 42  # Random Seed
    pretrained_model: str = 'beomi/kcbert-large'  # Transformers PLM name
    pretrained_tokenizer: str = ''  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    auto_batch_size: str = 'power'  # Let PyTorch Lightening find the best batch size
    batch_size: int = 0  # Optional, Train/Eval Batch Size. Overrides `auto_batch_size`
    lr: float = 5e-6  # Starting Learning Rate
    epochs: int = 3  # Max Epochs
    max_length: int = 150  # Max Length input size
    report_cycle: int = 100  # Report (Train Metrics) Cycle
    train_data_path: str = "./badword/ratings_train.csv"  # Train Dataset file
    val_data_path: str = "./badword/ratings_test.csv"  # Validation Dataset file
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    tpu_cores: int = 0  # Enable TPU with 1 core or 8 cores
