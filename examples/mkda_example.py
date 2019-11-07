"""Show an example of use of MKDA method."""

import pandas as pd
import nilearn
from nilearn import plotting, masking
import numpy as np

from neuroquery import datasets
from neuroquery.img_utils import get_masker, iter_coordinates_to_peaks_imgs
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
coordinates['weight'] = 1

time0 = time()
maps = Maps(
    coordinates,
    Ni=Ni,
    Nj=Nj,
    Nk=Nk,
    affine=affine,
    groupby_col='pmid',
    verbose=True
)
print(f'Loading df time {time()-time0}')

# r = 15
# n1 = affine[0, 0]
# n2 = affine[1, 1]
# n3 = affine[2, 2]
# kernel = _uniform_kernel(r, n1, n2, n3)

# maps.smooth(2, inplace=True, verbose=True)

time0 = time()
L = []
for x in maps.iter_imgs():
    L.append(x)
print(f'Iter time {time()-time0}')

time0 = time()
mkda_img = MKDA(L)
print(f'MKDA time {time()-time0}')
# plotting.view_img(mkda_img, threshold=3.0).open_in_browser()
# mkda_img = MKDA(iter_coordinates_to_peaks_imgs(coordinates))


plotting.view_img(mkda_img, threshold=3.0).open_in_browser()
#
