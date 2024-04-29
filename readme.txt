Workflow is to extract traces from tiff files using the workflow outlined in MatLabTrace Extractions

Then, use DO, AO, and some number of FRET samples with suitable characteristics to calculate alpha, beta, gamma, and delta parameters using the corresponding Python scripts in this archive. 

To apply these correction parameters to ebFRET formatted traces, use ApplyAlexCorr_toSelected.py. The assumed format will make use of only the donor-window excitation traces if ALEX experimental data is used, which are also those processed by the ebFRET wrapper. If no acceptor excitation window is used, only alph and gamma corrections are applied.