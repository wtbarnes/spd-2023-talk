# spd-2023-talk

Materials for my talk on MOXSI hot plasma modeling at 2023 SPD Meeting

## Simulations

To run the simulation pipeline with snakemake,

```
snakemake /output/dir/overlappogram_o{-4,-3,-2,-1,0,1,2,3,4}.fits --config client_address="<scheduler_address>" spectral_data_dir=/path/to/top/level_dir/spec/data results_dir=/path/to/where/to/results/go --cores 1
```

## Slides