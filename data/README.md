# Data Placement

The public repository does not include the raw OpenBMI MATLAB file because the teaching copy is large and should be managed as an external dataset dependency.

To reproduce the EEG section, place the provided dataset under the project root using the same layout:

```text
实验三-教学使用/
  s3/
    sess01_subj03_EEG_MI.mat
  gigascience_8_5_giz002.pdf
  实验三 - 时频分析.docx
```

The scripts locate the `.mat` file recursively under `实验三-教学使用/`. The repository includes derived CSV metrics and English figures so the reported outputs can be inspected without committing the raw dataset.
