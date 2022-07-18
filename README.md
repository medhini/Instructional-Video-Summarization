# Instructional-Video-Summarization
This is the official pytorch implementation of "TL;DW? Summarizing Instructional Videos with Task Relevance &amp; Cross-Modal Saliency" ECCV 2022.
In this repository we provide the datasets, code for pseudo ground-truth summary generation, training, and testing as described in the paper. 

If you find our repo useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{lin2022learning,
  title={TL;DW? Summarizing Instructional Videos with Task Relevance &amp; Cross-Modal Saliency},
  author={Narasimhan, Medhini and Nagrani, Arsha and Sun, Chen and Rubinstein, Michael and Darrell, Trevor and Rohrbach, Anna and Schmid, Cordelia},
  journal={ECCV},
  year={2022}
}
```

## Environment Setup

Create the conda environment from the yaml file and activate the environment,

```
conda env create -f vsum.yml
conda activate vsum
```

## Dataset

### WikiHow Summaries

First, extract the files in datasets/how_to_steps.tar.gz

```
tar -xzvf datasets/how_to_steps.tar.gz
```

Next, download the WikiHow Summaries dataset to the datasets folder by running

```
cd Instructional-Video-Summarization
python wikihow_video_download.py
```

## Pseudo Ground-Truth Summary Generation

## IV-Sum Training and Evaluation



