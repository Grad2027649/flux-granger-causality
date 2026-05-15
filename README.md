# Determining Significant Climate Drivers using Statistical Methods
This repository contains codes and data for the Granger-Causality test that was performed at individual Fluxnet/Ameriflux towers.

The CODES folder contains Python scripts that resample flux tower data (downloaded in csv files from https://fluxnet.org and https://ameriflux.lbl.gov) to the temporal resolutions for NDVI (16-day), SIF (on the 15th and end of the month), and VOD (10-day).  The code calling the **statsmodel** Python module for Granger-Causality test calculations can also be found in this folder.  The code for running the Linear Regression on two stations, as well as the Livezey-Chen Field Significance Test, are also included.

The DATA folder contains the resampled .xlsx station files for both the flux tower sites and the corresponding ERA5 coordinates over the same locations, which are the basis for running the univariate Granger-Causality test using the scripts in the CODES folder.

More data and codes will be available as the project and repository are continuously developed.
