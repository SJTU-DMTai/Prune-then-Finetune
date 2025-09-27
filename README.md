<h1 align="center"> Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2505.23195">
        <img
            src="https://img.shields.io/static/v1?label=arXiv&message=2505.23195&color=B31B1B&logo=arXiv"
            alt="arxiv"
        />
    </a>
    <!-- <a href="https://img.shields.io/github/stars/SJTU-Quant/Prune-then-Finetune">
        <img
            src="https://img.shields.io/github/stars/SJTU-Quant/Prune-then-Finetune"
            alt="Stars"
        />
    </a> -->
</p>

> This repo is a preview version and has not been fully tested yet. Feel free to create any issue!

## Updates
:triangular_flag_on_post: (2025.09) Our paper has been accepted at NeurIPS 2025!

:triangular_flag_on_post: (2025.05) Our preprint is made available at [ArXiv](https://arxiv.org/abs/2505.23195).

## Example

First, all datasets can be download from [Google Drive](https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322?usp=drive_link), and the dataset path can be specified by `--root_path ./dataset/`.

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
