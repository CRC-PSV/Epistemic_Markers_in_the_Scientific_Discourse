"""Various util functions"""
import random, shutil


def make_sub_corpus(dest_path, target_path, n_docs):
    """Copies a sample of files from one folder to another. Useful to build a sub-corpus for faster testing"""

    files = [f for f in dest_path.iterdir()]
    for f in random.sample(files, n_docs):
        shutil.copy(f, target_path / f.name)