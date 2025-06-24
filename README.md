# ICE
This is the official PyTorch codes for the paper:  
[**ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models**](https://arxiv.org/abs/2503.19902)  
[Fernando Julio Cendra](https://fcendra.github.io) and
[Kai Han](https://www.kaihan.org/)  
[Visual AI Lab](https://visailab.github.io/), The University of Hong Kong  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025  
<a href="https://visual-ai.github.io/ice/"><img alt='page' src="https://img.shields.io/badge/Project-Page-f3f6f9"></a>
<a href="https://arxiv.org/abs/2503.19902"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2503.19902-f3f6f9.svg"></a>

<p align="center">
<img src='assets/teaser.gif' width="90%">
</p>

## Installation
The environment can be installed through [conda](https://docs.conda.io/projects/miniconda/en/latest/) and pip. After cloning this repository, run the following command:
```shell
$ conda create -n ice python=3.11.11
$ conda activate ice

$ pip install -r requirements.txt
$ conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
$ pip install --upgrade keras-cv==0.6.4 tensorflow==2.14.0 numpy==1.23.5
$ pip uninstall tensorboard # TODO: need to fix this in the future
```

*After setting up the environment, it’s recommended to restart the kernel.

## Data & Setup
 Please ensure that:
 1. You create a folder (```$folder_name```) under the ```data/``` directory.
 2. Your input image is renamed to ```img.jpg``` before running this script, 
 
 thus the image will be located at ```data/$folder_name/img.jpg```

## Run concept extraction
The ICE framework operates through a two-stage process, *i.e*, Stage One: Automatic Concept Localization and Stage Two: Structured Concept Learning.

#### &rarr; **Stage One**: Automatic Concept Localization (please refer to [README-stage-one.md](doc/README-stage-one.md) for more details)
```shell
$ CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_one.sh
```
#### &rarr; **Stage Two**: Structured Concept Learning (please refer to [README-stage-two.md](doc/README-stage-two.md) for more details)
```shell
$ CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_two.sh
```

## Inference
(please refer to [README-infer.md](doc/README-infer.md) for more details)
```shell
$ CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh
```

## License

This project is under the CC BY-NC-SA 4.0 license. See [LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/) for details.

## Acknowledgements
Our code is developed based on [Break-A-Scene](https://github.com/google/break-a-scene).

## Citation
<span id="jump"></span>
```bibtex
@inproceedings{cendra2025ICE,
    author    = {Fernando Julio Cendra and Kai Han},
    title     = {ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```