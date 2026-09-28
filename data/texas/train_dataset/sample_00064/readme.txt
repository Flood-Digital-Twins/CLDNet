{
    "grid_attr": {
        "area": "[np.float64(225.0), 'km^2']",
        "shape": "(500, 500)",
        "cellsize": "[30.0, 'm']",
        "num_cells": 250000,
        "extent": {
            "left": 40895.86142907334,
            "right": 55895.86142907334,
            "bottom": 3430414.034508147,
            "top": 3445414.034508147
        }
    },
    "model_attr": {
        "case_folder": "data/texas/train_dataset/sample_00064",
        "birthday": "2025-09-06 14:27",
        "num_GPU": 1,
        "run_time": "[0, 172800, 900, 1800]",
        "num_gauges": 2
    },
    "initial_attr": {
        "h0": 0.0,
        "hU0x": 0,
        "hU0y": 0
    },
    "boundary_attr": {
        "num_boundary": 1,
        "boundary_details": "['0. (outline) fall, h and hU fixed as zero, number of cells: 1996']"
    },
    "rain_attr": {
        "num_source": 1,
        "max": "[np.float64(122.95), 'mm/h']",
        "sum": "[np.float64(222.28), 'mm']",
        "average": "[np.float64(4.63), 'mm/h']",
        "spatial_res": "[np.float64(15000.0), 'm']",
        "temporal_res": "[np.float64(3600.0), 's']"
    },
    "params_attr": {
        "manning": {
            "param_value": [
                0.035
            ],
            "land_value": [
                0
            ],
            "default_value": 0.035
        },
        "sewer_sink": 0,
        "cumulative_depth": 0,
        "hydraulic_conductivity": 0,
        "capillary_head": 0,
        "water_content_diff": 0
    }
}