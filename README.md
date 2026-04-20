# LSTM Streamflow Prediction in the Uinta Basin
### Comparing Standard LSTM and Entity-Aware LSTM for Spatial Generalization

**Jackson Hammond & Ben Kadunic** — CVEEN 5920 Machine Learning for Civil Computing, University of Utah

---

## Overview

This project implements and compares two LSTM-based neural network architectures for daily streamflow prediction across eight snow-dominated basins in the Uinta Mountains of northeastern Utah. The central question is whether incorporating static catchment attributes — drainage area, mean elevation, slope, and land cover — into an Entity-Aware LSTM (EA-LSTM) improves spatial generalization to an ungauged basin compared to a standard LSTM trained on meteorological forcing data alone.

Both models are evaluated using a spatial leave-one-out design: the Provo River basin (USGS 10154200) is withheld entirely from training and used as the spatial generalization test site.

---

## Results

| Model | NSE | RMSE (cms) | MAE (cms) |
|---|---|---|---|
| Baseline LSTM | 0.388 | 7.727 | 3.336 |
| EA-LSTM | 0.670 | 5.679 | 2.520 |

The EA-LSTM improved NSE by 0.28 and reduced RMSE by approximately 27% on the held-out test site. Both models successfully captured seasonal streamflow timing but struggled with extreme peak flow events, which are underrepresented in the training data.

---

## Study Sites

| USGS Gage ID | River | Area (km²) | Mean Elevation (m) | Role |
|---|---|---|---|---|
| 10154200 | Provo River | — | — | Spatial test (held out) |
| 10128500 | Weber River | 735.8 | 2756 | Train / Val |
| 10011500 | Bear River | 797.5 | 2958 | Train / Val |
| 09217900 | Blacks Fork | 586.3 | 3178 | Train / Val |
| 09266500 | Ashley Creek | 460.5 | 2879 | Train / Val |
| 09299500 | Whiterocks River | 529.4 | 3127 | Train / Val |
| 09292000 | Yellowstone River | 473.3 | 3260 | Train / Val |
| 09289500 | Lake Fork River | 355.7 | 3280 | Train / Val |

---

## Repository Structure

```
Machine-Learning-Project/
├── LSTM/
│   ├── notebooks/
│   │   ├── Baseline_LSTM.ipynb       # Standard LSTM training and evaluation
│   │   └── EALSTM_streamflow.ipynb   # EA-LSTM training and evaluation
│   ├── scripts/
│   │   ├── met_data_collection.py    # Daymet meteorological data
│   │   ├── snotel_collection.py      # NRCS SNOTEL SWE data
│   │   ├── streamflow_collection.py  # USGS NWIS streamflow data
│   │   ├── basin_attributes.py       # Topographic and land cover attributes
│   │   └── merge_data.py             # Merge all sources into one DataFrame
│   └── utils/
│       ├── LSTM_helper.py            # Standard LSTM model and helpers
│       └── EALSTM_helper.py          # EA-LSTM model and helpers
├── HW3env.yml                        # Conda environment
└── README.md
```

---

## Model Architecture

### Standard LSTM (Baseline)
A standard two-layer LSTM with 64 hidden units trained on eight dynamic meteorological features and observed SWE. All gates are computed from dynamic inputs at every timestep. The model has no access to basin physical characteristics.

### Entity-Aware LSTM (EA-LSTM)
Based on Kratzert et al. (2019). Extends the standard LSTM with a separate entity-aware input gate computed once per basin from static catchment attributes:

```
i = σ(W_i · x_static)        # input gate — fixed per basin
c_t = f_t ⊙ c_{t-1} + i ⊙ g_t   # entity-aware cell state update
```

This allows the model to learn a different internal sensitivity for each basin based on its physical characteristics, enabling better scaling of predictions to unseen basins.

---

## Data Sources

- **Meteorology:** Daymet v4 daily gridded climate data via `pydaymet` (1980–2024)
- **SWE:** NRCS SNOTEL station observations via [egagli/snotel_ccss_stations](https://github.com/egagli/snotel_ccss_stations)
- **Streamflow:** USGS NWIS daily discharge via `dataretrieval`
- **Basin attributes:** 30m DEMs via `py3dep`, NLCD 2019 land cover via `pygeohydro`

---

## Setup

```bash
conda env create -f HW3env.yml
conda activate HW3env
```

Run data collection scripts in order from the `LSTM/scripts/` directory:

```bash
python met_data_collection.py
python snotel_collection.py
python streamflow_collection.py
python basin_attributes.py
python merge_data.py
```

Then open the notebooks in `LSTM/notebooks/` to train and evaluate the models.

---

## References

Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22, 6005–6022.

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *Hydrology and Earth System Sciences*, 23, 5089–5110.

Gagliano, E. (2024). snotel_ccss_stations (Version v1.0) [Computer software]. https://github.com/egagli/snotel_ccss_stations
