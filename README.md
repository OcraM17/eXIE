# Explaining Image Enhancement Black-Box Methods through a Path Planning Based Algorithm

Official Repo of the paper "Explaining Image Enhancement Black-Box Methods through a Path Planning Based Algorithm". At the moment the paper is
**under review**.

[Marco Cotogni](https://scholar.google.com/citations?user=8PUz5lAAAAAJ&hl=it) and [Claudio Cusano](https://scholar.google.com/citations?hl=it&user=lhZpU_8AAAAJ&view_op=list_works&sortby=pubdate)

[![arxiv](https://img.shields.io/badge/arXiv-red)](https://arxiv.org/pdf/2207.07092.pdf)

eXIE is an algorithm for explaining the results of state-of-the-art image-to-image
translation methods, used for natural image enhancement. Despite their high
accuracy, these methods often suffer from limitations such as artifact generation
and scalability to high resolutions, and have a completely black-box approach
that does not provide any insight into the enhancement processes applied.

eXIE solves this issue by providing a step-by-step explanation of the output
produced by existing enhancement methods, using a variant of the A* algorithm to
emulate the enhancement process through the application of equivalent enhancing
operators.

This algorithm has been applied to several state-of-the-art models trained on
the Five-K dataset, and has produced sequences of enhancing operators that
produce results that are very similar in terms of performance and overcome the
poor interpretability of the best-performing algorithms.

<p align="center">
<img src="imgs/eXie.png" width="400" height="auto"/>
</p>

## Requirements
python > 3.7, PyTorch, Torchvision, PIL, numpy

## Datasets
Download the [Five-K](https://data.csail.mit.edu/graphics/fivek/) datasets.
Once the dataset has been downloaded, split the data in training (4000)
and test(1000) using the files train1+2-list.txt and test-list.txt

## Training
Train one of your favorite image enhancement methods