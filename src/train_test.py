from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import random
import pandas as pd


@hydra.main(version_base=None, config_path="../cfg", config_name="main")
def main(cfg):
    random.seed(cfg.random.seed)

    """
    Dataset containing pmids from:
    - Google Scholar (GS): negative samples, golden standard, expert curated
    - LitSuggest (LS): negative samples, silver standard, not curated
    - Cellosaurus (CS): positive samples, gold standard, expert curated
    """
    data_raw = {k: open(v, "r").read().splitlines() for k, v in cfg.data.raw.items()}
    logging.info({k: len(v) for k, v in data_raw.items()})

    """
    Test data
    - test negatives contain all PMIDs from GS
    - test positives are sampled from CS
    
    Two test_set variants:
    - test_50_50: contains an equal sample of positives and negatives.
    - test_80_20: contains a majority sample (80%) of negatives. 
    """

    # test_50_50
    test_neg = [{"PMID": pmid, "LABEL": 0} for pmid in data_raw["gs_neg_pmid"]]
    logging.info(f"test_neg {len(test_neg)}")
    test_50_50_pos = [
        {"PMID": pmid, "LABEL": 1}
        for pmid in random.sample(
            list(set(data_raw["cs_pos_pmid"]).difference(data_raw["gs_neg_pmid"])),
            len(test_neg),
        )
    ]
    logging.info(f"test_50_50_pos {len(test_50_50_pos)}")
    test_50_50 = pd.DataFrame.from_dict(test_neg + test_50_50_pos, orient="columns")
    test_50_50.to_csv(cfg.data.processed.test_50_50, sep="\t", index=False)

    # test_80_20
    test_80_20_pos = random.sample(
        test_50_50_pos, round((1 / (0.8 / 0.2)) * len(test_50_50_pos))
    )  # subset of test_50_50_pos
    logging.info(f"test_80_20_pos {len(test_80_20_pos)}")
    test_80_20 = pd.DataFrame.from_dict(test_neg + test_80_20_pos, orient="columns")
    test_80_20.to_csv(cfg.data.processed.test_80_20, sep="\t", index=False)

    """
    Train data
    - negatives; all PMIDs from ls_neg_pmid, not already sampled for test
    - positives; all PMIDs from cs_pos_pmid not already sampled for test
    """
    train_neg = [
        {"PMID": pmid, "LABEL": 0}
        for pmid in set(data_raw["ls_neg_pmid"]).difference(test_50_50["PMID"])
    ]
    logging.info(f"train_neg {len(train_neg)}")
    train_pos = [
        {"PMID": pmid, "LABEL": 1}
        for pmid in set(data_raw["cs_pos_pmid"]).difference(test_50_50["PMID"])
    ]
    logging.info(f"train_pos {len (train_pos)}")
    train = pd.DataFrame.from_dict(train_neg + train_pos, orient="columns")
    train.to_csv(cfg.data.processed.train, sep="\t", index=False)


if __name__ == "__main__":
    main()
