# CoT-T5
Config files for CoT on T5X

## Preperations

to cache the dataset, `export PYTHONPATH="<path>/CoT-T5/"`
use the following
```
seqio_cache_tasks \
   --tasks=.* \
   --output_cache_dir="/fsx/lintangsutawika/data/" \
   --module_import=cot_t5.mixtures
```

If you have difficulty downloading open files from gcp
```
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export TENSORSTORE_CA_BUNDLE="/etc/ssl/certs/ca-bundle.crt"
```