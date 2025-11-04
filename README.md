# Image Segmentation Toolkit

A modular image segmentation toolkit that combines **classical computer vision** and **deep learning** approaches.  
The toolkit provides an interactive **[Gradio](https://www.gradio.app/)** interface for comparing multiple segmentation algorithms on any image.

---

## Features

### Classical Segmentation Methods
- **[Otsu's Thresholding](https://ieeexplore.ieee.org/document/4310076)** — automatic global thresholding for binary segmentation  
- **K-means Clustering** — color-based segmentation with adjustable clusters  
- **[SLIC Superpixels](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/)** — region-based segmentation preserving edges  
- **[Watershed Algorithm](https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html)** — gradient-based separation of touching objects  
- **[Felzenszwalb’s Graph-Based Segmentation](https://cs.brown.edu/~pff/segment/)** — adaptive graph-based method for region merging  

### Deep Learning Models
- **[SegNet](https://arxiv.org/abs/1511.00561)** with **[EfficientNet-B0](https://arxiv.org/abs/1905.11946)** backbone  
- **SegNet with VGG backbone** for architecture comparison  

### Ensemble Methods
- **Otsu + SegNet** — merges classical boundaries with semantic labels  
- **Custom Ensemble Segmentation** — adjustable fusion of deep and classical methods  

---

## Installation

### Prerequisites
- Python ≥ 3.8  
- [PyTorch ≥ 1.10](https://pytorch.org/get-started/locally/)  
- CUDA-compatible GPU *(optional but recommended)*  

### Setup
```bash
# Clone the repository
git clone https://github.com/themaverick/Image_Segmentation.git
cd Image_Segmentation

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python download_models.py
```

## Project Structure
```
IMAGE_SEGMENTATION/
├── app.py                      # Main application with  pretrained models
├── experiments/                # Implementation of segmentation algorithms
│   ├── ensemble_method.py      # Ensemble segmentation implementation
│   ├── felzenszwalb_segmentation/ # Felzenszwalb algorithm implementation
│   ├── kmeans_segmenter.py     # K-means segmentation implementation
│   ├── enhanced_kmeans_segmenter.py # SLIC implementation
│   ├── otsu_segmenter.py       # Otsu thresholding implementation
│   ├── watershed_segmenter.py  # Watershed algorithm implementation
│   └── SegNet/                 # Deep learning models
│       ├── efficient_b0_backbone/ # EfficientNet backbone for SegNet
│       └── vgg_backbone/       # VGG backbone for SegNet
├── saved_models/              # Directory for pretrained weights
└── requirements.txt           # Package dependencies
