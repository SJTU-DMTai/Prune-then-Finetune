<h2 align="center"> Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning</h2>

<p align="center">
    <a href="https://arxiv.org/abs/2505.23195">
        <img
            src="https://img.shields.io/static/v1?label=arXiv&message=2505.23195&color=B31B1B&logo=arXiv"
            alt="arxiv"
        />
    </a>
</p>

## Updates
:triangular_flag_on_post: (2025.09) Our paper has been accepted at NeurIPS 2025!
:triangular_flag_on_post: (2025.05) Our preprint is made available at [ArXiv](https://arxiv.org/abs/2505.23195).

## Example

Full Finetune:
```bash
sh example/post-training/finetune/Chronos.sh "ETTh1 ETTh2 ETTm1 ETTm2 weather"
sh example/post-training/finetune/test/Chronos.sh "ETTh1 ETTh2 ETTm1 ETTm2 weather"
```

Prune-then-Finetune:
```bash
sh example/post-training/prune-then-finetune/prune/Chronos.sh "ETTh1 ETTh2 ETTm1 ETTm2 weather"
sh example/post-training/prune-then-finetune/finetune/Chronos.sh "ETTh1 ETTh2 ETTm1 ETTm2 weather"
sh example/post-training/prune-then-finetune/test/Chronos.sh "ETTh1 ETTh2 ETTm1 ETTm2 weather"
```

All scripts could be found in `example/benchmark/`.


## Citation
If you find our code helpful, please consider citing our paper:
```
@inproceedings{
less-is-more-prune-then-finetune,
title={Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning},
author={Lifan Zhao and Yanyan Shen and Zhaoyang Liu and Xue Wang and Jiaji Deng},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=jy4bBsr1Jc}
}
```