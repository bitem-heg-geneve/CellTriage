# CellTriage
The project aims at using state-of-the-art machine learning methods, and in particular transformers, to develop a dedicated triage service to help Cellosaurus curators to navigate the literature. The project’s main goal is improving scalability of Cellosaurus curation to gain efficiency and comprehensiveness.

## Data
- data/cs_pos_pmid_set.tsv; curated positive samples, extracted from cellosaurus, 22719 pmids
- data/gs_neg_pmid.tsv; curated negative samples, extracted from Google Scholar, 475 pmids
- data/ls_neg_pmid.tsv; uncurated negative samples, extract from LitSuggest, 645 pmids