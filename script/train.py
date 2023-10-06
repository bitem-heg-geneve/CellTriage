import hydra
from ct_model import CtDataModule, CtModel
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


from transformers import (
    AutoTokenizer,
)

from pytorch_lightning.loggers import TensorBoardLogger


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings


warnings.simplefilter(action="ignore", category=FutureWarning)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


@hydra.main(version_base=None, config_path="../cfg", config_name="main")
def main(cfg):
    pl.seed_everything(cfg.random.seed)

    for dataset, v in cfg.checkpoint.items():
        TRAIN_FP = cfg.data.processed.text[dataset]
        LABELS = cfg.model.labels
        VALIDATION_RATIO = cfg.model.validation_ratio

        df = pd.read_json(TRAIN_FP, lines=True)
        train_df, val_df = train_test_split(
            df,
            test_size=VALIDATION_RATIO,
            random_state=cfg.random.seed,
            stratify=df[LABELS],
        )
        for text_col, v in cfg.checkpoint[dataset].items():
            TEXT_COL = cfg.model.text_col[text_col]
            for lm, ckpt in cfg.checkpoint[dataset][text_col].items():
                LM_MODEL_NAME = cfg.model.lm[lm]
                tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
                MAX_TOKEN_COUNT = cfg.model.max_token_count
                BATCH_SIZE = cfg.model.batch_size
                N_EPOCH = cfg.model.n_epoch
                WARMUP_RATIO = cfg.model.warmup_ratio
                PATIENCE = cfg.model.patience
                MODEL_VARIANT_NAME = f"{dataset}_{text_col}_{lm}"
                LOGGER_NAME = MODEL_VARIANT_NAME
                CHECKPOINT_DIRPATH = (
                    cfg.model.checkpoint_dirpath + "/" + MODEL_VARIANT_NAME
                )

                data_module = CtDataModule(
                    train_df,
                    val_df,
                    text_col=TEXT_COL,
                    tokenizer=tokenizer,
                    labels=LABELS,
                    batch_size=BATCH_SIZE,
                    max_token_len=MAX_TOKEN_COUNT,
                )

                steps_per_epoch = len(train_df) // BATCH_SIZE
                total_training_steps = steps_per_epoch * N_EPOCH
                warmup_steps = round(total_training_steps * WARMUP_RATIO)

                model = CtModel(
                    lm_model_name=LM_MODEL_NAME,
                    labels=LABELS,
                    n_warmup_steps=warmup_steps,
                    n_training_steps=total_training_steps,
                )

                checkpoint_callback = ModelCheckpoint(
                    dirpath=CHECKPOINT_DIRPATH,
                    filename="best-checkpoint",
                    save_top_k=1,
                    verbose=True,
                    monitor="val_loss",
                    mode="min",
                )

                logger = TensorBoardLogger("lightning_logs", name=LOGGER_NAME)

                early_stopping_callback = EarlyStopping(
                    monitor="val_loss", patience=PATIENCE
                )

                trainer = pl.Trainer(
                    logger=logger,
                    callbacks=[early_stopping_callback, checkpoint_callback],
                    max_epochs=N_EPOCH,
                    accelerator="auto",
                )

                trainer.fit(model, data_module)
                del model


if __name__ == "__main__":
    main()
