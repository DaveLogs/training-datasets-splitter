# training-dataset-splitter

Split the training datasets into 'training'/'validation'/'test' data.


## Installation system requirements
Install the python packages.

```bash
(venv) $ pip install -r requirements.txt
```


## Usage example

```bash
(venv) $ python split.py \
    --input_path ./input \
    --output_path ./output \
    --group_name group1 \
    --split_name training-validation-test \
    --split_ratio 0.8-0.1-0.1
```


## Data Structures

The structure of data folder as below.

### Input data

```
/input
├── /group1
│   #   [filename].[ext]
│   ├── image_00001.jpg
│   ├── image_00002.jpg
│   ├── ...
│   └── labels.txt
│   
├── /group2
└── ...
```

* Label 'labels.txt' file structure:

```
    {filename}\t{label}\n
    image_00001.jpg    abcd
    image_00002.jpg    efgh
    ...
```


### Output data

```
/output
├── /training
│   ├── gt.txt
│   └── /images
│	    #   [filename].[ext]
│	    ├── image_00001.jpg
│	    ├── image_00001.jpg
│	    └── ...
│
├── /validation
└── /test
```

* Ground truth 'gt.txt' file structure:

```
#   {filename}\t{label}\n
    images/image_00001.jpg    abcd
    images/image_00002.jpg    efgh
    ...
```
