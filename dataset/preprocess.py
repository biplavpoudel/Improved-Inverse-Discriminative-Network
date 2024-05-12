from PIL import Image
import os
import pandas as pd
import random

size = 55
num_genuine = 24
num_forged = 24


def resize_img(root, w=220, h=115):
    os.mkdir(f'{root}_resize')
    for filename in os.listdir(root):
        with Image.open(f'{root}/{filename}') as img:
            img = img.resize((w, h))
            img.save(f'{root}_resize/{filename}')


def pair_string_genuine(i, j, k):
    return f'full_org_resize/original_{i}_{j}.png full_org_resize/original_{i}_{k}.png 1\n'


def pair_string_forged(i, j, k):
    return f'full_org_resize/original_{i}_{j}.png full_forg_resize/forgeries_{i}_{k}.png 0\n'


def generate(file, i):
    # reference-genuine pairs
    for j in range(1, num_genuine + 1):
        for k in range(j + 1, num_genuine + 1):
            file.write(pair_string_genuine(i, j, k))
    # reference-forged pairs
    org_forg = [(j, k) for j in range(1, num_genuine + 1)
                for k in range(1, num_forged + 1)]
    for (j, k) in random.choices(org_forg, k=276):
        file.write(pair_string_forged(i, j, k))

    '''
    Generate reference-test pair for dataset.

    Input:
        root: path of dataset
        mode: which dataset
        cutting_point: the cutting point to split dataset into train and test
    Output:
        None
    '''


def generate_pairs(root: str, cutting_point: int):
    with open(f'{root}/train_pairs.txt', 'w') as f:
        for i in range(1, cutting_point):
            generate(f, i)

    with open(f'{root}/test_pairs.txt', 'w') as f:
        for i in range(cutting_point, size + 1):
            generate(f, i)


if __name__ == '__main__':
    resize_img(r'D:\MLProjects\Inverse-Discriminative-Network\dataset\CEDAR\signatures\full_org')
    resize_img(r'D:\MLProjects\Inverse-Discriminative-Network\dataset\CEDAR\signatures\full_forg')
    generate_pairs(r'D:\MLProjects\Inverse-Discriminative-Network\dataset\CEDAR\signatures', 51)

