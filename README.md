# Determining Significant Climate Drivers using Statistical Methods
This repository contains codes and data for the Granger-Causality test that was performed at individual Fluxnet/Ameriflux towers.

The CODES folder contains Python scripts that resample flux tower data (downloaded in csv files from https://fluxnet.org) to the temporal resolutions for NDVI (16-day), SIF (on the 15th and end of the month), and VOD (10-day).

The DATA folder contains the resampled .xlsx station files, which are the basis for running the univariate Granger-Causality test using the scripts in the CODES folder.
