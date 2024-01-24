# Continuous Learning Object Classification

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Continuous Learning Object Classification is a Python-based system developed for the MEng Graduation Project sponsored by Wakeb_Data. The project addresses catastrophic forgetting in deep neural networks by implementing a self-learning system capable of rapid adaptation. This is achieved by training on targeted 5-second video data for new classes, eliminating the need for full dataset retraining and enhancing efficiency and adaptability.

## Demo

Watch a demo of the project: [Demo Video](Demo.mkv)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Implementation](#implementation)
- [Final Status and Goal Achievement](#final-status-and-goal-achievement)
- [Project Management](#project-management)
- [License](#license)

## Installation

To experiment with SLDA on CORe50, run:

```bash
$ jupyter notebook test-slda.ipynb
```

To run the deployment and use the interface, execute:

```bash
$ python vision3.py
```


## Usage

The project provides two main functionalities through the following files:
- `test-slda.ipynb`: Experiment with SLDA on CORe50 dataset.
- `vision3.py`: Run deployment and use the interface.

## Implementation

### Programming Languages and Technologies

The project was developed using Python, with data handling and computational tasks managed through Kaggle and Google Colab. PyCharm served as the integrated development environment for coding and debugging. Key libraries/tools include PyTorch, cv2, Streamlit, and Avalanche.

### Development Environment

The development environment combined Kaggle, Google Colab, and PyCharm, creating a versatile platform that supported the project’s diverse needs.

### Implementation Process

The implementation process included several essential steps:
1. Preparation of the CORe50 dataset.
2. Application of transformations using PyTorch’s transforms module.
3. Setup of DataLoader instances for batch processing.
4. Integration of the ResNet-18 model with the Streaming Linear Discriminant Analysis (SLDA) model for feature extraction and incremental learning.
5. Incremental training of the hybrid model, improving its accuracy with each new batch of data.

## Final Status and Goal Achievement

The project achieved an 80% accuracy rate in object classification, effectively updating the model with new classes. This milestone demonstrated the efficacy of the project and was positively received by our sponsor, Wakeb_Data.

## Project Management

Employing an Agile methodology, the project maintained a flexible and iterative approach to development. Regular feedback sessions from the sponsor and mentor guided the project’s evolution, proving instrumental in navigating the challenges of continual learning and model adaptation.

## Citation

If you find this project useful for your work, please consider citing the following paper, which inspired the model used in this project:

```
@InProceedings{Hayes_2020_CVPR_Workshops,
    author = {Hayes, Tyler L. and Kanan, Christopher},
    title = {Lifelong Machine Learning With Deep Streaming Linear Discriminant Analysis},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2020}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
