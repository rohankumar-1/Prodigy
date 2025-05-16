

## Regarding the transition from Python 3.6 to Python 3.9:

The main changes occur in `anomaly_detector.py`, `vae.py`, and `data_pipeline.py`. These changes migrate the Tensorflow code into PyTorch 2.6.0. Another important update was to `tsfresh`, which is an integral part of Prodigy's design. The `tsfresh` version that was compatible with Python 3.6 had a dependency called `matrixprofile`, which is no longer supported and does not correctly function with Python 3.9. Therefore, in the reproducibility experiments, the preselected features found in `fe_eclipse_tsfresh_raw_CHI_2000.json` need to be reinspected such that features that depend on `matrixprofile` are removed.

To accomplish this, I started writing `clean_features.ipynb`, with the goal to reselect valid features, and furthermore run the reproducibility experiments with success. I expect slightly worse results as fewer features will be available during training and inference, but this could be improved by reselecting a new set of 2000 features altogether. 

There are some other parts that I have not yet tested fully, such as loading data in the data pipeline or creating windows. 

