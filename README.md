# VALSE :dancer:

:dancer: VALSE: A Task-Independent Benchmark for Vision and Language Models Centered on Linguistic Phenomena. https://arxiv.org/abs/2112.07566

## Data Instructions
Please find the data in the `data` folder. The dataset is in `json` format and contains the following relevant fields:
* A reference to the image in the original dataset: `dataset` and `image_file`.
* The valid sentence, the caption for VALSE: `caption`.
* The altered caption, the `foil`.
* The annotator's votes (3 annotators per sample): `mturk`.
    * The subentry `caption` counts the number of annotators who chose the caption, but/and not the foil, to be the one describing the image.
    * The subentry `foil` counts how many of the three annotators chose the foil to be (also) describing the image.
    * For more information, see subsec. 4.4 and App. E of the [paper](https://arxiv.org/abs/2112.07566).

:bangbang: Please be aware that the jsons are containing both valid (meaning: validated by annotators) and non-validated samples. In order to work only with the **valid set**, please consider filtering them:

> We consider a **valid foil** to mean: at least two out of three annotators identified the caption, but not the foil, as the text which accurately describes the image.

This means that the valid samples of the dataset are the ones where `sample["mturk"]["caption"] >= 2`.

Example instance:
```python
{
    "actions_test_0": {
        "dataset": "SWiG",
        "original_split": "test",                 # the split of the original dataset in which the sample belonged to
        "dataset_idx": "exercising_255.jpg",      # the sample id in the original dataset
        "linguistic_phenomena": "actions",        # the linguistic phenomenon targeted
        "image_file": "exercising_255.jpg",
        "caption": "A man exercises his torso.",
        "classes": "man",                         # the word of the caption that was replaced
        "classes_foil": "torso",                  # the foil word / phrase
        "mturk": {
            "foil": 0,
            "caption": 3,
            "other": 0
        },
        "foil": "A torso exercises for a man."
    }, ...
}
```

## Images
For the images, please follow the downloading instructions of the respective original dataset. The provenance of the original images is mentioned in the json files in the field `dataset`.

## Reference
Please cite our [:dancer: VALSE paper](https://arxiv.org/abs/2112.07566) if you are using this dataset.

```
@misc{parcalabescu2021valse,
      title={VALSE: A Task-Independent Benchmark for Vision and Language Models Centered on Linguistic Phenomena}, 
      author={Letitia Parcalabescu and Michele Cafagna and Lilitta Muradjan and Anette Frank and Iacer Calixto and Albert Gatt},
      year={2021},
      eprint={2112.07566},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
