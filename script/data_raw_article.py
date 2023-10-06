import aiohttp
import asyncio
import hydra
import pandas as pd
from random import randint
from orjson import loads, dumps
import logging
import os
from munch import Munch


async def fetchSibils(session, url, ids, col):
    params = {"ids": ",".join(map(str, ids)), "col": col}
    async with session.post(url + "/fetch", params=params) as response:
        r = await response.read()
    res = loads(r.decode("utf-8"))
    if "sibils_article_set" in res.keys():
        logging.info(f"fetched {len(res['sibils_article_set'])} articles from {col}")
        res_dict = {}
        for item in res["sibils_article_set"]:
            id = item.pop("_id")
            res_dict[id] = item
        return res_dict
    else:
        return {}


async def getArticle(sem, session, url, pmids, article_path):
    async with sem:
        # MEDLINE fetch
        ml_res = await fetchSibils(session, url, pmids, col="medline")
        pmcids = []
        articles = []
        for a in ml_res.values():
            pmcid = a["document"]["pmcid"]
            if pmcid != "":
                pmcids.append(pmcid)
            article = {
                "PMID": a["document"]["pmid"],
                "PMCID": pmcid,
                "ABSTRACT": a["document"]["title"] + " " + a["document"]["abstract"],
                "FULLTEXT": None,
            }
            articles.append(article)

        # PMC fetch
        if len(pmcids) > 0:
            pmc_res = await fetchSibils(session, url, pmcids, "pmc")
            for article in articles:
                pmcid = article["PMCID"]
                if pmcid in pmc_res.keys():
                    art_pmc = pmc_res[pmcid]
                    article["FULLTEXT"] = " ".join(
                        [s["sentence"] for s in art_pmc["sentences"]]
                    )
                else:
                    article["FULLTEXT"] = None

        # write articles to file
        for article in articles:
            with open(
                (os.path.join(article_path, article["PMID"] + ".json")), "wb"
            ) as f:
                f.write(dumps(article))
        logging.info(f"written {len(articles)} articles to {article_path}")


async def run(cfg):
    one_week_pmids = pd.read_csv("data/celltriage/oneweek.csv")["PMID"].tolist()
    dataset = Munch()
    for k, v in cfg.data.processed.pmid.items():
        dataset[k] = pd.read_json(v, lines=True)

    dataset_combined = pd.concat([dataset[d] for d in dataset], ignore_index=True)
    dataset_pmids = dataset_combined["PMID"].unique().tolist()

    pmid_set = set(one_week_pmids + dataset_pmids)

    ARTICLE_REPO_PATH = cfg.data.raw.article
    pmid_in_repo = [
        int(".".join(f.split(".")[:-1])) for f in os.listdir(ARTICLE_REPO_PATH)
    ]

    pmid_set = list(pmid_set.difference(pmid_in_repo))
    print(f"{len(pmid_set)} PMIDs")

    batches = (
        pmid_set[i : i + cfg.sibils.batch_size]
        for i in range(0, len(pmid_set), cfg.sibils.batch_size)
    )

    connector = aiohttp.TCPConnector(limit_per_host=cfg.sibils.limit_per_host)
    sem = asyncio.Semaphore(cfg.sibils.semaphore_size)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            getArticle(sem, session, cfg.sibils.url, b, cfg.data.raw.article)
            for b in batches
        ]  # create list of tasks
        await asyncio.gather(*tasks)  # execute them in concurrent manner


@hydra.main(version_base=None, config_path="../cfg", config_name="main")
def main(cfg):
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
