# CellTriage
The project aims at using state-of-the-art machine learning methods, and in particular transformers, to develop a dedicated triage service to help Cellosaurus curators to navigate the literature. The project’s main goal is improving scalability of Cellosaurus curation to gain efficiency and comprehensiveness.

## Data
### Raw
- cs_term_45.0; cell line terms, extracted from Cellosaurus, 145673 terms
#### PMID
- Cellosaurus; positive samples, extracted from Cellosaurus, 22719 PMIDs
- CellosaurusAB; positive samples, extracted from Cellosaurus, curated, high portion of seminal papers, 10.000 PMIDs
- GoogleScholar; negative samples, extracted from rejected Google Scholar results, curated, 509 PMIDs
- LitSuggest; negative samples, extract from rejected LitSuggest results, 645 PMIDs