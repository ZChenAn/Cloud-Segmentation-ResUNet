# Cloud Segmentation Using Machine Learning

Author: TianYi Zhang
Dissertation Submission Date: 2023/8/24

This repository contains the implementation of the Cloud Segmentation method as described in TianYi Zhang's dissertation titled "Cloud Segmentation Using Machine Learning." The project includes preprocessing of images, training, validation, rectification and testing phases for cloud segmentation using a ResNet50-based U-Net model.

## Prerequisites

- GPU: 3060 Ti, dual-core GPU computer（My own usage）
- Nvidia Cuda version 11.8
- cuDNN version 8.9
- Python (the project is implemented using Python, a widely used, straightforward, advanced, and general-purpose programming language)
- Libraries: Pytorch, numpy, opencv-python, scikit-image, matplotlib, Pillow

## Dataset

Our dataset is now only used for private testing and is not available to the public, you can see some of our images in the demo's sample. If the dataset is published, you should put all our dataset images in the folder path Cloud-Segmentation-ResUNet\dataset\test\webcam\images.
Or if you have your own sky/cloud dataset, you can just put it in the folder path I just mentioned and our model will output the results in the path \dataset\test\pred_full.

## Project Structure

The code is organized into separate Python scripts for modularity and easier debugging:

- `main.py`: Coordinates the flow of the project.

- `ResUNet.py`: Contains the implementation of the ResNet50-based U-Net model.

- `rectify.py`: Responsible for the rectification strategy.

- `handle_images.py`: Contains utility functions for handling images.

- `test.py`: Handles the UCL Webcam dataset for testing.

- `demo.ipynb`: A Jupyter notebook for streamlined execution and visualization.

- `ty_rectify.pth`: the pre-trained model of this project.

  download link: https://drive.google.com/file/d/16L3Zn9Q5yUDxohhhXRldxozVmfG2z2OP/view?usp=sharing
## How to Run

### For Quick Testing

1. Open the `demo.ipynb` Jupyter notebook.
2. Download and use the pre-trained model `ty_rectify.pth`. It should be in the same folder with `demo.ipynb`.
3. Run the cells to test the sky images and visualize the results.

### For Full Execution

1. Ensure that the dataset is placed in the correct directories (Cloud-Segmentation-ResUNet\dataset\test\webcam\images).
2. Run `main.py` to coordinate the entire flow of the project, including training, validation, and testing.

## Citation

If you find this code useful in your research, please cite the following:

```bibtex
@article{zhang2023cloud,
  title={Cloud Segmentation Using Machine Learning},
  author={Zhang, TianYi},
  year={2023},
  publisher={School of Physics and Astronomy of University College London}
}
```

## License

Please refer to the LICENSE file in this repository for the licensing information.

## Updates

The code and repository will be maintained and updated regularly. Feel free to contribute or report issues.

