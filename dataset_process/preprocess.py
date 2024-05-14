from PIL import Image
import os
from utils import parse_args
import random

size = 55
num_genuine = 24
num_forged = 24
args = parse_args()


def resize_img(root, w=220, h=115):
    if not os.path.exists(f'{root}_resize'):
        os.mkdir(f'{root}_resize')
    for filename in os.listdir(root):
        if filename.endswith('.db'):
            continue
        else:
            with Image.open(f'{root}/{filename}') as img:
                img = img.resize((w, h))
                img.save(f'{root}_resize/{filename}')


def genuine_pair(i, j, k):
    return f'full_org_resize/original_{i}_{j}.png full_org_resize/original_{i}_{k}.png 1\n'


def forged_pair(i, j, k):
    return f'full_org_resize/original_{i}_{j}.png full_forg_resize/forgeries_{i}_{k}.png 0\n'


def generate(file, i):
    # reference-genuine pairs
    for j in range(1, num_genuine + 1):
        for k in range(j + 1, num_genuine + 1):
            file.write(genuine_pair(i, j, k))
    # reference-forged pairs
    org_forg = [(j, k) for j in range(1, num_genuine + 1)
                for k in range(1, num_forged + 1)]
    for (j, k) in random.choices(org_forg, k=276):
        file.write(forged_pair(i, j, k))


def generate_pairs(root: str, split: int):
    # Generate reference-test pair for dataset_process.
    with open(f'{root}/train_pairs.txt', 'w') as f:
        for i in range(1, split):
            generate(f, i)

    with open(f'{root}/test_pairs.txt', 'w') as f:
        for i in range(split, size + 1):
            generate(f, i)


if __name__ == '__main__':
    resize_img(r'D:\MLProjects\Inverse-Discriminative-Network\dataset_process\CEDAR\signatures\full_forg')
    resize_img(r'D:\MLProjects\Inverse-Discriminative-Network\dataset_process\CEDAR\signatures\full_org')
    generate_pairs(args.pairs_path, 51)

