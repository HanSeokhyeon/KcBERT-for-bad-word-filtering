import torch

from pytorch_lightning import Trainer, seed_everything

from config.arg_badword_labeled import Arg
from model import Model

args = Arg()

# args.tpu_cores = 8  # Enables TPU
args.fp16 = True  # Enables GPU FP16
args.batch_size = 16  # Force setup batch_size


def main():
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)
    seed_everything(args.random_seed)
    model = Model(args)

    print(":: Start Training ::")
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        auto_scale_batch_size=args.auto_batch_size if args.auto_batch_size and not args.batch_size else False,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
        # For TPU Setup
        tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    trainer.fit(model)

    model.save_model()
    model.upload_model()


if __name__ == '__main__':
    main()
