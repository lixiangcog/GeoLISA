# GeoLISA

GeoLISA is a continual test-time adaptation method for medical image
segmentation. It combines language-induced semantic anchors with
distribution-geometry feature alignment while keeping the source segmenter
frozen during deployment.

The work is currently under submission. The repository currently provides the
OD/OC segmentation implementation.

## Installation

```bash
conda create -n geolisa python=3.10 -y
conda activate geolisa
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-runtime.txt
```

## Data

Download and extract the fundus dataset:

```bash
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/data_lx/Fundus.zip
unzip Fundus.zip -d data/Fundus
```

The data directory should contain the domain folders and CSV splits for
RIM-ONE-r3, REFUGE, ORIGA, REFUGE-Valid, and Drishti-GS.

## Usage

Prepare the SAM ViT-B checkpoint and BioMedCLIP text embeddings under
`models/`, then set the following paths:

```bash
export GEOLISA_ENV=/absolute/path/to/geolisa/environment
export FUNDUS_ROOT=/absolute/path/to/data/Fundus
bash LISA_OPTIC.sh
```

Source training and continual adaptation can also be submitted with
`scripts/train_source.slurm` and `scripts/run_geolisa.slurm`.

## Acknowledgements

This implementation builds on
[VPTTA](https://github.com/Chen-Ziyang/VPTTA),
[Segment Anything](https://github.com/facebookresearch/segment-anything), and
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

## License

This project is released under the [MIT License](LICENSE).
