# Data for *lil*Gym

This repository contains the data used in [lilGym: Natural Language Visual Reasoning with Reinforcement Learning (Wu et al., 2022)](https://arxiv.org/abs/2211.01994).

## Structure of the files

Each of the four configurations (`TowerScratch`, `TowerFlipIt`, `ScatterScratch`, `ScatterFlipIt`) has three splits (training, development and test). 

For instance, the training data corresponding to the configuration `TowerScratch` can be found in `tower_scratch_train.json`.

### JSON files



#### Scratch

Each of the JSON files for the Scratch configurations is structured as below:

```
{
    "sentence_id": {
        "sentence": ...,
        "lf": ...,
    },
}
```

`sentence_id` is a unique id for the sentence.

The fields are:

- `sentence` (str): a human-written natural language sentence.
- `lf` (str): the logical form (Python program) we collected, that corresponds to the sentence.

#### FlipIt

Each of the JSON files for the FlipIt configurations is structured as below:

```
{
    "sentence_id": {
        "sentence": ...,
        "lf": ...,
        "label": ...,
        "structured_rep": ...,
    },
}
```

`sentence_id` is a unique id for the sentence.

The fields are:

- `sentence` (str): a human-written natural language sentence.
- `lf` (str): the logical form (Python program) we collected, that corresponds to the sentence.
- `label` (str): the label for the example, "true" or "false". Note that this label *does* match the image and the sentence, and it is always *flipped* to its reverse in the environment to obtain the target boolean.
- `structured_rep` (List[List[Dict]]): the structured representation of the image, which is a list of length three. For each item in this list, which represents a box, there is another list of items (up to length eight). For each item, there is an x and y position (x_loc and y_loc), a type (the name of the shape), a color, and a size. (From [NLVR](https://aclanthology.org/P17-2034/))
