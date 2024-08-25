Rough first draft of Python FRET analysis pipeline.

Workflow is to extract traces from tiff files using the workflow outlined in MatLabTrace Extractions

Then, use DO, AO, and some number of FRET samples with suitable characteristics to calculate alpha, beta, gamma, and delta parameters using the corresponding Python scripts in this archive. 

To apply these correction parameters to ebFRET formatted traces, use ApplyAlexCorr_toSelected.py. The assumed format will make use of only the donor-window excitation traces if ALEX experimental data is used, which are also those processed by the ebFRET wrapper. If no acceptor excitation window is used, only alpha and gamma corrections are applied.

In Progress: 
Migration to a purely Pythonic workflow instead of the old mixed MATLAB/Python workflow.
Because of a quirk of the image saving by the camera installed on the FRET1 microscope without metadata, images are occasionally saved in the incorrect order with respect to laser cycling for ALEX mode. To correct this, the first step is to reorder the images, and this is done based on blue/green excitation window thresholding since further red excitation should negligibly excite the dyes in the lower-wavelength channels. 
In the MATLAB approach, traces are extracted from re-mapped images (should double check this), whereas the Pythonic approach is to match peaks based on re-mapped images, but then to do ALL calculations with un-mapped images to maximally preserve raw data.
Python version includes more flexible thresholding methods and opens up possibility to more robust automated processing. 
Current implementation used in publication was MatLab/Python-mixed version modified for extraction of ALEX data. 
Currently correction parameter pipeline relies on mixed pipeline for initial processing and will be merged. 
Beadmapping relies on mapping by Python script. Currently trace extraction includes definition of polywarp function written to complement skimage's. SK-im warp is applied to images, but for matching of peaks this function operates directly on x, y coordinates so that peak-finding can be performed on raw images and order of peak coordinates maintained for easier indexing. 