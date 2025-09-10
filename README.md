# BeeTector 🐝

A deep model for Bombus subspecies detection

_A deep learning project for identifying Bumblebee subspecies from images, built at Le Wagon Data Science & AI Bootcamp._

## Overview

- **Problem:** To classify subspecies of bumblebee from an uploaded photo.

- **Why it matters:** Bumblebee conservation is extremely important due to their role in pollination.  Their populations are declining due to habitat loss,
climate change and pesticides.  This project aims to create the possibility to
track both their location and prevalence.

- **Solution:** We developed a two-stage system using InceptionV3: the first model detects whether an image contains a bumblebee, and the second classifies it into one of 12 subspecies. Both models are served via a FastAPI backend and containerised with Docker for deployment.

## Demo

[Streamlit demo](https://beetector.streamlit.app/)


## Project Structure

```bash
├── api
├── app.py
├── bee_tector
├── Dockerfile
├── docs
├── Makefile
├── models
├── notebooks
├── raw_data
├── README.md
├── requirements.txt
├── setup.py
└── tests
```

## Installation and Setup

```bash
git clone https://github.com/katherinestewart/bee-tector.git
cd bee-tector
```

#### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

## Data

We trained our models on subsets of the [The iNaturalist Species Classification and Detection Dataset](https://arxiv.org/abs/1707.06642?).

The `raw_data` (not included in repo) contains a CSV with the data on where and when the images were taken along with our training datasets.
- `bumble_ternary` with train, val and test for 3 classes; Bumblebees, lookalikes and others.
- `bombus12_full` with train, val and test sets for the 12 subspecies.

## Models

#### 3 Class Model
| Model          | Accuracy | Notes               |
|:---------------|:--------:|--------------------:|
| InceptionV3    |   92%    | First model         |

#### 12 Class Model
| Model          | Accuracy | Notes                 |
|:---------------|:--------:|----------------------:|
| Baseline CNN   |   26%    | First benchmark       |
| InceptionV3    |   81%    | Best performing model |


## Results
## Results

Our best results were achieved using **SGD with learning rate scheduling** (dropping the LR every 10 epochs), inspired by this [paper](https://www.nature.com/articles/s41598-021-87210-1) in Scientific Reports.

- **3-class model (InceptionV3)**: 92% accuracy
- **12-class model (InceptionV3)**: 81% accuracy (vs 26% baseline CNN)

Training curves below show the convergence behaviour:
![Training curves](raw_data/learning_curves.png)
Detailed results are shown in the [Models](#models) section.

## Team

- [Katherine Stewart](https://www.linkedin.com/in/katherine-stewart-a3933b354/)
- [Emanuele Torrisi](https://www.linkedin.com/in/emanuele-torrisi-08a3572a4/)
- [Rohan Raghava Poojary]()
- [Thahyra van Heyningen](https://www.linkedin.com/in/thahyravh/)
