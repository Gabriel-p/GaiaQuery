# GaiaQuery

Simple script to query Gaia DR2 data using the
[`astroquery`](https://astroquery.readthedocs.io) package. Generates two user-defined colors (not present in the raw Gaia data) with their associated uncertainties.

Can be made to apply the [Babusiaux et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..10G/abstract) quality filters.

Latest release can be obtained [here](https://github.com/Gabriel-p/GaiaQuery/releases).

### Requirements

    Python 3.7, numpy, astropy, matplotlib, scipy, astroquery, uncertainties

Can be run in a `conda` environment with:

    $ conda create -n gaiaq numpy astropy matplotlib scipy
    $ conda activate gaiaq
    $ conda install -c conda-forge uncertainties
    $ conda install -c astropy astroquery

![Alt text](out.png?raw=true)