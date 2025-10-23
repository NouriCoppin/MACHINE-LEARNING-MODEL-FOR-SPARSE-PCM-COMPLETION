# Data Notes

Expected CSV `source_id,target_id,label` with integer node ids and binary outcomes.
Use the included synthetic generator:

```bash
python -m src.utils.synth --n 200 --m 5000 --out data/ncaa_sample.csv
```
