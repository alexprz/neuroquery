import numpy as np
import pandas as pd

from nilearn import image, input_data
from nilearn.datasets import load_mni152_brain_mask

from meta_analysis import Maps


def get_masker(mask_img=None, target_affine=None):
    if isinstance(mask_img, input_data.NiftiMasker):
        return mask_img
    if mask_img is None:
        mask_img = load_mni152_brain_mask()
    if target_affine is not None:
        if np.ndim(target_affine) == 0:
            target_affine = np.eye(3) * target_affine
        elif np.ndim(target_affine) == 1:
            target_affine = np.diag(target_affine)
        mask_img = image.resample_img(
            mask_img, target_affine=target_affine, interpolation="nearest"
        )
    masker = input_data.NiftiMasker(mask_img=mask_img).fit()
    return masker


def coords_to_voxels(coords, ref_img=None):
    if ref_img is None:
        ref_img = load_mni152_brain_mask()
    affine = ref_img.affine
    coords = np.atleast_2d(coords)
    coords = np.hstack([coords, np.ones((len(coords), 1))])
    voxels = np.linalg.pinv(affine).dot(coords.T)[:-1].T
    voxels = voxels[(voxels >= 0).all(axis=1)]
    voxels = voxels[(voxels < ref_img.shape[:3]).all(axis=1)]
    voxels = np.floor(voxels).astype(int)
    return voxels


def coords_to_peaks_img(coords, mask_img):
    mask_img = image.load_img(mask_img)
    voxels = coords_to_voxels(coords, mask_img)
    peaks = np.zeros(mask_img.shape)
    np.add.at(peaks, tuple(voxels.T), 1.0)
    peaks_img = image.new_img_like(mask_img, peaks)
    return peaks_img


def coords_to_peaks_arr(coords, mask_img):
    mask_img = image.load_img(mask_img)
    voxels = coords_to_voxels(coords, mask_img)
    peaks = np.zeros(mask_img.shape)
    np.add.at(peaks, tuple(voxels.T), 1.0)
    return peaks


def gaussian_coord_smoothing(
    coords, mask_img=None, target_affine=None, fwhm=9.0
):
    masker = get_masker(mask_img, target_affine)
    peaks_img = coords_to_peaks_img(coords, mask_img=masker.mask_img_)
    img = image.smooth_img(peaks_img, fwhm=fwhm)
    return masker.inverse_transform(masker.transform(img).squeeze())


def coordinates_to_maps(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    print(
        "Transforming {} coordinates for {} articles".format(
            coordinates.shape[0], len(set(coordinates["pmid"]))
        )
    )
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    images, img_pmids = [], []
    for pmid, img in iter_coordinates_to_maps(
        coordinates, mask_img=masker, fwhm=fwhm
    ):
        images.append(masker.transform(img).ravel())
        img_pmids.append(pmid)
    return pd.DataFrame(images, index=img_pmids), masker


def coordinates_to_maps4(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    print(
        "Transforming {} coordinates for {} articles".format(
            coordinates.shape[0], len(set(coordinates["pmid"]))
        )
    )
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    images, img_pmids = [], []
    for pmid, img in iter_coordinates_to_maps2(
        coordinates, mask_img=masker, fwhm=fwhm
    ):
        images.append(masker.transform(img).ravel())
        img_pmids.append(pmid)
    return pd.DataFrame(images, index=img_pmids), masker


def iter_coordinates_to_maps(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    articles = coordinates.groupby("pmid")
    for i, (pmid, coord) in enumerate(articles):
        print(
            "{:.1%} pmid: {:< 20}".format(i / len(articles), pmid),
            end="\r",
            flush=True,
        )
        img = gaussian_coord_smoothing(
            coord.loc[:, ["x", "y", "z"]].values, fwhm=fwhm, mask_img=masker
        )
        yield pmid, img

def iter_coordinates_to_maps2(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    mask_img = masker.mask_img
    Ni, Nj, Nk = mask_img.shape
    print(Ni, Nj, Nk)
    affine = mask_img.affine
    coordinates['weight'] = 1
    maps = Maps(
        coordinates,
        Ni=Ni, Nj=Nj, Nk=Nk,
        affine=affine,
        groupby_col='pmid',
        verbose=True
        )
    articles = coordinates.groupby("pmid")
    sigma = 2*fwhm/(2.355*affine[0, 0])
    # maps.smooth(sigma, inplace=True, verbose=True)
    myiter = maps.iter_smooth_imgs(sigma)
    for i, (pmid, coord) in enumerate(articles):
        print(
            "{:.1%} pmid: {:< 20}".format(i / len(articles), pmid),
            end="\r",
            flush=True,
        )
        # img = gaussian_coord_smoothing(
        #     coord.loc[:, ["x", "y", "z"]].values, fwhm=fwhm, mask_img=masker
        # )
        img = next(myiter)
        yield pmid, img


def iter_coordinates_to_peaks_imgs(
    coordinates, mask_img=None, target_affine=(4, 4, 4)
):
    myiter = iter_coordinates_to_maps(
        coordinates, mask_img, target_affine, None
    )

    for _, img in myiter:
        yield img


def iter_coordinates_to_arrs(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    articles = coordinates.groupby("pmid")
    for i, (pmid, coord) in enumerate(articles):
        print(
            "{:.1%} pmid: {:< 20}".format(i / len(articles), pmid),
            end="\r",
            flush=True,
        )
        arr = coords_to_peaks_arr(
            coord.loc[:, ["x", "y", "z"]].values, mask_img=masker.mask_img_
        )
        yield arr
