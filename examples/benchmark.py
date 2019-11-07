import pandas as pd
import nilearn
from nilearn import plotting, masking
import numpy as np

from neuroquery import datasets
from neuroquery.img_utils import get_masker, iter_coordinates_to_peaks_imgs, iter_coordinates_to_arrs
from neuroquery.meta_analysis import MKDA

from meta_analysis import Maps

from time import time

template = nilearn.datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(template)
Ni, Nj, Nk = template.shape
affine = template.affine

masker = get_masker(mask_img=None, target_affine=(4, 4, 4))
Ni, Nj, Nk = masker.mask_img.shape
affine = masker.mask_img.affine

coordinates = pd.read_csv(datasets.fetch_peak_coordinates())


@profile
def benchmark1():
    """Benchmark 1: PR code"""
    L = []
    for x in iter_coordinates_to_peaks_imgs(coordinates):
        L.append(x)

    mkda_img = MKDA(L)
    plotting.view_img(mkda_img, threshold=3.0).open_in_browser()


@profile
def benchmark2():
    """Benchmark2: Using meta_analysis package."""
    coordinates['weight'] = 1

    target_affine = (4, 4, 4)
    masker = get_masker(mask_img=None, target_affine=target_affine)
    affine = masker.mask_img_.affine

    maps = Maps(
        coordinates,
        Ni=Ni,
        Nj=Nj,
        Nk=Nk,
        affine=affine,
        groupby_col='pmid',
        verbose=True
    )

    L = []
    for x in maps.iter_arrs():
        L.append(x)

    mkda_img = MKDA(L, affine=affine)
    plotting.view_img(mkda_img, threshold=3.0).open_in_browser()


@profile
def benchmark3():
    """Benchmark 3: Using improved version of PR code."""
    target_affine = (4, 4, 4)
    masker = get_masker(mask_img=None, target_affine=target_affine)
    affine = masker.mask_img_.affine

    L = []
    for x in iter_coordinates_to_arrs(coordinates, target_affine=target_affine):
        L.append(x)

    mkda_img = MKDA(L, affine=affine)
    plotting.view_img(mkda_img, threshold=3.0).open_in_browser()


# benchmark1()
benchmark2()
# benchmark3()
