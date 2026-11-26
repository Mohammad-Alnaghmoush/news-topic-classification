# News Topic Classification

This project classifies English headlines from AG News into four topics: World, Sports, Business, and Science and Technology. It compares TF-IDF and frozen MiniLM embeddings, each with Logistic Regression, on the same official test headlines. The short-headline subset contains titles with one to six words after cleaning.

[Open the project notebook](News_Topic_Classification.ipynb) · [Try the live demo](https://mohammad-alnaghmoush.github.io/news-topic-classification/)

## Live demo

Enter a headline or select one of the examples. Choose **TF-IDF**, **MiniLM**, or **Compare both** to see the predicted topic and category scores. Predictions run in your browser using the saved classifiers. TF-IDF also works offline once you have the page. MiniLM downloads its encoder the first time you select it.

To run the page locally, download or clone this repository and open `index.html` in a recent browser. Keep the model and manifest files in the same directory if you want to use the download links.

## Test results

| Method | Test subset | Headlines | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | Full | 7,600 | 86.87% | 86.84% |
| TF-IDF + Logistic Regression | Short | 3,643 | 84.60% | 84.45% |
| Frozen MiniLM + Logistic Regression | Full | 7,600 | 83.49% | 83.46% |
| Frozen MiniLM + Logistic Regression | Short | 3,643 | 80.54% | 80.49% |

The notebook contains the data checks, development scores, confusion matrices, error analysis, and per-class metrics. Its saved tables and figures are visible before running cells.

## Download the fitted models

| File | Contents |
| --- | --- |
| [`tfidf_logistic_regression.pkl.gz`](tfidf_logistic_regression.pkl.gz) | Fitted TF-IDF vectorizer and fitted Logistic Regression pipeline |
| [`minilm_logistic_regression.pkl.gz`](minilm_logistic_regression.pkl.gz) | Fitted Logistic Regression classifier for normalized 384-dimensional MiniLM vectors |
| [`model_manifest.json`](model_manifest.json) | SHA-256 checksums, label order, and the exact MiniLM encoder revision |

The two checkpoints use Python's gzipped pickle format. Use the package versions in `requirements.txt` to load these files in Python. The MiniLM classifier also needs the [pinned encoder][encoder]. Use the encoder to produce L2-normalized embeddings before calling the classifier.

## Reproduce the experiment

Open the notebook in Jupyter Notebook or Google Colab. For a local run, use a Python 3.11 environment. From the repository folder, install the dependencies and launch the notebook:

```sh
python -m pip install -r requirements.txt
python -m notebook News_Topic_Classification.ipynb
```

For a CPU-only PyTorch installation, install `torch==2.6.0` from the [official CPU wheel index](https://download.pytorch.org/whl/cpu) before installing the requirements.

Run the notebook cells in order from a fresh session. The first full run downloads the checksum-verified AG News archive and the pinned encoder. Training and embedding generation can take several minutes. The setup cell prints the output directory for generated data, checkpoints, figures, and metrics.

## Data and encoder

- AG News labels, titles, and descriptions come from the authors' version 3 distribution. The notebook records the source and checksums. See the archive README for dataset usage terms. [AG News source](https://github.com/zhangxiangxiao/Crepe)
- MiniLM uses `sentence-transformers/all-MiniLM-L6-v2` at the revision listed in the manifest. [Encoder model card][encoder]

[encoder]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
