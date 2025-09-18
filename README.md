# IR-Spectroscopy-denoising
spectral_denoise/
├── data/
│   ├── raw/                # Place your .txt files here
│   ├── processed/          # For saving outputs later
│   └── endpoints.csv       # Endpoint data file
├── models/
│   ├── baseline_ae.h5      # Will be saved after training
│   └── resunet_model.h5   # Will be saved after training
├── scripts/
│   ├── load_qc.py         # Data loading and QC
│   ├── baseline.py        # ALS baseline removal
│   ├── baseline_ae.py    # Shallow autoencoder baseline
│   ├── resunet_model.py  # 1D ResUNet architecture
│   ├── augment_data.py   # Functions for Sim2Real augmentation
│   ├── metrics.py        # Evaluation metrics
│   └── train_resunet.py # Training the ResUNet
├── notebooks/
│   └── demo_analysis.ipynb # Interactive notebook for plots and testing
└── README.md              # Instructions and overview
