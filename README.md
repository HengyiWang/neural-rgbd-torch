# neural-rgbd-torch
This project is a PyTorch implementation of [Neural RGB-D Surface Reconstruction](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/static/pdf/neural_rgbd_surface_reconstruction.pdf), which is a novel approach for 3D reconstruction that combines implicit surface representations with neural radiance fields

## Installation

```
git clone https://github.com/HengyiWang/neural-rgbd-torch
cd neural-rgbd-torch
pip install -r requirements.txt
```

Please also install the external Marching cube packages via:

```
cd external/NumpyMarchingCubes
python setup.py install
```



  ## Dataset

The ScanNet dataset can be downloaded via the following link [neural_rgbd_data](http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip). The ICL data can be downloaded from the original author's [webpage](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)

## Run

```
python optimize.py --config configs/<config_file>.txt

```

## Citation
Thanks for the author for their amazing works:
```
@InProceedings{Azinovic_2022_CVPR,
    author    = {Azinovi\'c, Dejan and Martin-Brualla, Ricardo and Goldman, Dan B and Nie{\ss}ner, Matthias and Thies, Justus},
    title     = {Neural RGB-D Surface Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {6290-6301}
}
```
