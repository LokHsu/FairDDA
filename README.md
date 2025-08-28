# FairDDA

This repository contains source codes and datasets for our **CIKM'25** paper:

- Improving Recommendation Fairness via Graph Structure and Representation Augmentation

## Usage
### Pre-Train

FairDDA uses pre-trained LightGCN embeddings as input. The pre-trained model parameters are available from the [FairMI](https://github.com/chenzhao-hfut/FairMI) repository.

### Train & Test

- Training FairDDA on MovieLens:
```shell
python main.py --dataset=ml-1m
```

- Training FairDDA on LastFM:
```shell
python main.py --dataset=lastfm-360k --batch_size=4096 --ub_reg=40
```

- Training FairDDA on IJCAI15:
```shell
python main.py --dataset=ijcai --num_epochs=70 --sigma=0.5 --reconn_reg=0.1 --lb_reg=0.05 --ub_reg=100
```

## Additional Results

Due to space limitations, the ACM version only analyzes and reports recommendation performance under gender fairness  (binary attribute scenario). For additional experiments and analysis in **multi-class sensitive scenarios**, please refer to Appendix A of our [arXiv version](https://arxiv.org/abs/2508.19547).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{FairDDA,
    title = {Improving Recommendation Fairness via Graph Structure and Representation Augmentation},
    author = {Xu, Tongxin and Liu, Wenqiang and Bin, Chenzhong and Xiao, Cihan and Zeng, Zhixin and Gu, Tianlong},
    booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
    year = {2025}
}
```
