# Run your own MSA server with ColabFold

mmseqs is software that searches for MSAs very quickly, it is a pretty low level API.
ColabFold provides a higher level API for protein folding use cases.

## Run
Will take ~3 hours:
`modal run modal_setup_databases.py`

Will take 20 minutes:
`modal run run_colabfold_search`


## References
- [ColabFold](https://github.com/sokrypton/ColabFold)
- [mmseqs](https://github.com/soedinglab/MMseqs2)


## Instructions
- [ColabFold MSA Setup](https://github.com/sokrypton/ColabFold?tab=readme-ov-file#generating-msas-for-large-scale-structurecomplex-predictions)
- [mmseqs setup](https://github.com/soedinglab/mmseqs2/wiki#compile-from-source-under-linux) 
  

## Status
- Was hitting
  [this](https://github.com/soedinglab/MMseqs2/issues/616#issuecomment-1507286459) but
seems to have gone away after redownloading everything onto a fresh volume.
- Now hitting `Input /vol/data/colabfold_envdb_202108_db_seq does not exist` even though
  the file is read earlier in the script... [modal logs](https://modal.com/apps/modal-labs/examples/deployed/example-colabfold-search?start=1736970656&end=1737057056&live=true&activeTab=logs)


## Other useful info:
- Wiki: https://github.com/sokrypton/ColabFold/wiki
- [Rough idea of colabfold_search](https://github.com/sokrypton/ColabFold/blob/main/colabfold_search.sh)
- [mmseqs user guide](https://mmseqs.com/latest/userguide.pdf)
