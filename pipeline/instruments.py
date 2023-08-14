"""
MOXSI instrument class for producing DEMs at MOXSI plate scale and resolution
"""
import astropy.units as u
from astropy.wcs.utils import pixel_to_pixel
import dask
import dask.array
import ndcube
import numpy as np

from mocksipipeline.detector.response import convolve_with_response
from mocksipipeline.util import read_data_cube
from overlappy.util import strided_array


def sample_spectral_cube(lam, channel, instr_cube_wcs, observer):
    """
    Sample a Poisson distribution based on counts from spectral
    cube and map counts to detector pixels.

    Parameters
    ----------
    lam
        Poisson expectation in units of photons. Should be 4D 
        with dimensions corresponding to time, wavelength,
        and space.
    channel
    instr_cube_wcs
    observer
    """
    samples = dask.array.random.poisson(lam=lam, size=lam.shape).sum(axis=0).compute()
    idx_nonzero = np.where(samples>0)
    weights = samples[idx_nonzero].astype(float)
    # Weights are in photons; want to convert them to ct
    ct_per_photon = channel.electron_per_photon * channel.camera_gain
    # NOTE: we can select the relevant conversion factors this way because the wavelength
    # axis of lam is the same as channel.wavelength and thus their indices are aligned
    weights *= ct_per_photon.to_value('ct / ph')[idx_nonzero[0]]
    # Map counts to detector coordinates
    overlap_wcs = channel.get_wcs(observer)
    idx_nonzero_overlap = pixel_to_pixel(instr_cube_wcs, overlap_wcs, *idx_nonzero[::-1])
    n_rows = channel.detector_shape[0]
    n_cols = channel.detector_shape[1]
    hist, _, _ = np.histogram2d(idx_nonzero_overlap[1], idx_nonzero_overlap[0],
                                bins=(n_rows, n_cols),
                                range=([-.5, n_rows-.5], [-.5, n_cols-.5]),
                                weights=weights)
    return ndcube.NDCube(strided_array(hist, overlap_wcs.array_shape[0]),
                         wcs=overlap_wcs,
                         unit='ct')


@dask.delayed
def calculate_expectation(time_index, spec_cube_dir, channel):
    """
    This saves a timestep of the spectral cube in photons for a
    particular spectral order to a single index in a Zarr array
    so that a Dask array can easily be created from it.
    """
    spec_cube = read_data_cube(spec_cube_dir / f'spec_cube_t{time_index}.fits', hdu=1, use_fitsio=True)
    instr_cube = convolve_with_response(spec_cube, channel, electrons=False)
    return (instr_cube * (1*u.pix)).to('photon / s').data
