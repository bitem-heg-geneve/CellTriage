# CellTriage
The project aims at using state-of-the-art machine learning methods, and in particular transformers, to develop a dedicated triage service to help Cellosaurus curators to navigate the literature. The project’s main goal is improving scalability of Cellosaurus curation to gain efficiency and comprehensiveness.

## Data
### Raw
- cs_45.json; cell line terms, extracted from Cellosaurus; 145673 terms
- cs_pos_pmid_set.tsv; curated positive samples, extracted from Cellosaurus, 22719 pmids
- gs_neg_pmid.tsv; curated negative samples, extracted from Google Scholar, 475 pmids
- ls_neg_pmid.tsv; uncurated negative samples, extract from LitSuggest, 645 pmids

