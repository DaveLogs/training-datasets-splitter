"""

Split the training datasets into 'training'/'validation'/'test' data.

## Example usage:
    python split.py \
        --input_path ./input \
        --output_path ./output \
        --split_name training-validation-test \
        --split_ratio 0.8-0.1-0.1

## Data structure:

### Input data
/input
#   [filename].[ext]
├── image_00001.jpg
├── image_00002.jpg
├── ...
└── labels.txt

* Label 'labels.txt' file structure:
#   {filename}\t{label}\n
    image_00001.jpg    abcd
    image_00002.jpg    efgh
    ...

### Output data
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

*  Ground truth 'gt.txt' file structure:
#   {filename}\t{label}\n
    images/image_00001.jpg    abcd
    images/image_00002.jpg    efgh
    ...

"""

import os
import sys
import argparse
import random
import shutil

from sklearn.model_selection import train_test_split


def run(args):
    """ Split the training datasets """

    # split info configuration
    args.split_name = args.split_name.split('-')
    args.split_ratio = [float(ratio) for ratio in args.split_ratio.split('-')]
    validation_of_split_group_info(args.split_name, args.split_ratio)

    # validation of raw datasets
    files, labels, count = validation_of_raw_datasets(args.input_path)

    # create output directory
    if os.path.isdir(args.output_path):
        sys.exit(f"'{args.output_path}' directory is already exists.")
    else:
        split_group_path = create_working_directory(args.output_path, args.split_name)

    # split the training datasets
    split_group = split_training_datasets(files, args.split_name, args.split_ratio)

    # move data and create 'gt.txt'
    for _, key in enumerate(split_group):
        digits = len(str(len(split_group[key])))
        gt_file = open(os.path.join(split_group_path[key], "gt.txt"), "w", encoding="utf8")
        images_path = os.path.join(split_group_path[key], "images")
        os.makedirs(images_path)

        print("split group: ", key)
        for idx, item in enumerate(split_group[key]):
            if (idx + 1) % 100 == 0:
                print(("\r%{}d / %{}d Processing !!".format(digits, digits)) % (idx + 1, len(split_group[key])), end="")

            filename = os.path.join(images_path, os.path.basename(item))
            gt = labels[os.path.basename(item)]

            gt_file.write('%s\t%s\n' % (os.path.join("images", os.path.basename(item)), gt))
            shutil.copy(item, filename)

        gt_file.close()
        print("\n")

    # summary of splitted datasets
    print("-" * 50)
    print("Total datasets count: ", count)
    for _, key in enumerate(split_group):
        print("'%s' group's data count: %d" % (key, len(split_group[key])))

    return


def validation_of_split_group_info(name, ratio):
    """ validation of split group info """
    if len(name) != len(ratio):
        sys.exit(f"split_name list '{name}' and split_ratio list '{ratio}' are not same.")
    elif len(name) <= 1 or len(name) > 3:
        sys.exit(f"Invalid split count.")


def validation_of_raw_datasets(path):
    """ validation of raw datasets """

    # output path
    if not os.path.isdir(path):
        sys.exit(f"Can't find '{path}' directory.")
    files, count = get_files(path, except_file="labels.txt")

    # labels file
    labels = {}
    labels_file_path = os.path.join(path, 'labels.txt')
    if not os.path.isfile(labels_file_path):
        sys.exit(f"Can't find '{labels_file_path}' file.")
    with open(labels_file_path, 'r') as l:
        label_list = l.readlines()
        for _, line in enumerate(label_list):
            filename, label = line.strip('\n').split('\t')
            labels[filename] = label

    # Compare the number of files with the number of labels.
    if count != len(labels):
        sys.exit("Number of files and number of labels are not the same.")

    return files, labels, count


def get_files(path, except_file=""):
    file_list = []
    abspath = os.path.abspath(path)

    for file in os.listdir(path):
        if file.startswith(".") or file == except_file:
            # print('except file name: ', file)
            continue

        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


def create_working_directory(root, sub_dirs):
    sub_dirs_list = {}
    os.makedirs(root)
    for sub in sub_dirs:
        path = os.path.join(root, sub)
        sub_dirs_list[sub] = path
        os.makedirs(path)

    return sub_dirs_list


def split_training_datasets(files, name_list, ratio_list):
    """ split the training datasets """
    group_data = {}

    random.shuffle(files)
    group_data[name_list[0]], group_data[name_list[1]] = \
        train_test_split(files, train_size=ratio_list[0])

    if len(name_list) == 3:
        group_data[name_list[1]], group_data[name_list[2]] = \
            train_test_split(group_data[name_list[1]], train_size=ratio_list[1]/(ratio_list[1]+ratio_list[2]))

    # for idx in range(0, len(name_list)):
    #     print(f"args.split_name[{idx}]: ", len(group_data[name_list[idx]]))

    return group_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split the training datasets into training/validation/test data.")

    parser.add_argument("--input_path", type=str, required=True, help="Path of the raw datasets")
    parser.add_argument("--output_path", type=str, required=True, help="Path of the splitted datasets")
    parser.add_argument("--split_name", type=str, default="training-test", help="Name of each group to be split")
    parser.add_argument("--split_ratio", type=str, default="0.9-0.1", help="Assign ratio for each group to be split")

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    arguments = parse_arguments()
    run(arguments)
