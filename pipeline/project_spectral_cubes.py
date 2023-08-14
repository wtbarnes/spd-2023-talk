import pathlib

from astropy.wcs.utils import wcs_to_celestial_frame
import dask.array
import distributed

from mocksipipeline.util import read_data_cube
from mocksipipeline.detector.response import convolve_with_response, SpectrogramChannel
from overlappy.io import write_overlappogram

from instruments import calculate_expectation, sample_spectral_cube


client = distributed.Client(address=snakemake.config['client_address'])
channel = SpectrogramChannel(int(snakemake.params.spectral_order), full_detector=False)
time_interval = float(snakemake.config['time_interval'])

# Get list of spectral cube files
spectra_output_dir = pathlib.Path(snakemake.input[0])
spec_cube_files = list(spectra_output_dir.glob('spec_cube_t*.fits'))
n_time = len(spec_cube_files)

# NOTE: precomputing one instrument cube here to get the WCS and needed dimensions
tmp_spec_cube = read_data_cube(spec_cube_files[0], hdu=1, use_fitsio=True)
tmp_instr_cube = convolve_with_response(tmp_spec_cube, channel, electrons=False,)
instr_cube_wcs = tmp_instr_cube.wcs
shape = tmp_instr_cube.data.shape
dtype = tmp_instr_cube.data.dtype

lam = dask.array.stack(
    [dask.array.from_delayed(calculate_expectation(i, spectra_output_dir, channel), shape, dtype)
     for i in range(n_time)],
    axis=0,
) * time_interval

# Map sampled photons to detector
observer = wcs_to_celestial_frame(tmp_spec_cube.wcs).observer
overlappogram = sample_spectral_cube(lam, channel, instr_cube_wcs, observer)

# Update metadata
for k in ['OBSRVTRY', 'TELESCOP', 'CHANNAME']:
    overlappogram.meta[k] = tmp_instr_cube.meta[k]

# Write to FITS file
pathlib.Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
write_overlappogram(overlappogram, snakemake.output[0])
