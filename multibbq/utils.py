"""Inference-path text utilities.

Dataset-generation helpers live in notebooks/utils.py next to the notebooks
that use them.
"""
import re


def add_in_the_image(context: str, nonstereotype_group_name: str, stereotype_group_name: str) -> str:
    specific_nouns = ['man', 'woman', 'person', 'student','manager','employee','guy','kid','girl','boy','teen','teenager','children','people','friend','neighbor',]
    groups = [g for g in [nonstereotype_group_name, stereotype_group_name] if g]

    if not groups:
        return context

    gpat = '|'.join(sorted(map(re.escape, groups), key=len, reverse=True))
    npat = '|'.join(map(re.escape, specific_nouns))

    # 1) "group + noun"（not following 'in the image'）
    pattern1 = rf'\b(?:{gpat})\s+(?:{npat})\b(?!\s+in the image)'
    out = re.sub(pattern1, r'\g<0> in the image', context)

    # 2) "group"：not following noun and 'in the image'
    pattern2 = rf'\b(?:{gpat})\b(?!\s+(?:{npat})\b)(?!\s+in the image)'
    out = re.sub(pattern2, r'\g<0> in the image', out)

    return out
