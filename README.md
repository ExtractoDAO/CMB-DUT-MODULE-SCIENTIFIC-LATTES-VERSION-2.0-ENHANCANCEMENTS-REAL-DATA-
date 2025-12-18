# DUT-CMB Cosmological Modeling Web Service

FastAPI web service for running and visualizing Dead Universe Theory (DUT) cosmological simulations with CMB constraints and global observational data.

## 🚀 Quick Start

### Prerequisites

- Python >= 3.9
- pip (Python package manager)

### Installation

1. **Save the original DUT module**

   Save the provided DUT code as `dut_module_engine.py` in your project directory.

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web service**

   ```bash
   python main.py
   ```

   Or using uvicorn directly:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the interface**

   Open your browser and navigate to:

   ```
   http://localhost:8000
   ```

## 📡 API Endpoints

### Web Interface

- `GET /` - Main interactive dashboard

### API Endpoints

- `POST /api/simulate` - Run simulation with default or custom parameters
- `GET /api/cache` - Retrieve last simulation results
- `GET /api/data-sources` - View global data sources
- `GET /api/health` - Health check endpoint

## 🧪 Example API Usage

### Run simulation with default parameters:

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Run simulation with custom parameters:

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "H0_kms_mpc": 68.0,
    "omega_b": 0.05,
    "omega_c": 0.26,
    "lambda_phi": 0.12
  }'
```

### Get cached results:

```bash
curl http://localhost:8000/api/cache
```

## 📊 Features

- **Interactive Visualizations**: Real-time plotting of H(z) and fσ₈(z)
- **CMB Observables**: Computation of lA, r_s, and thermalization redshift
- **Global Data Integration**: Pantheon+, Planck, DESI, and more
- **Unit Testing**: Automatic validation of CMB acoustic scale
- **JSON Export**: Download complete simulation results
- **Responsive Design**: Modern, dark-themed interface

## 🔬 Scientific Background

This module implements the Dead Universe Theory (DUT) cosmological framework with:

- Background evolution with scalar field φ
- Linear perturbations and growth factors
- CMB distance priors (acoustic scale lA)
- Integration with Pantheon+ SNIa data
- Support for BAO, H(z), and fσ₈ constraints

### Citation Required

If you use this code in research or publications, please cite:

```
Almeida, J. (2025).
Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
Hubble and Growth Tensions Simultaneously:
Δχ² = –211.6 with Identical Datasets.
Zenodo. https://doi.org/10.5281/zenodo.17752029
```

## 📁 Project Structure

```
.
├── dut_module_engine.py    # Original DUT-CMB scientific module
├── main.py                 # FastAPI web service
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .dut_cache/            # Created automatically for data caching
    ├── PantheonPlus_SH0ES.dat
    ├── PantheonPlus_SH0ES.meta.json
    └── ...
```

## 🛠️ Development

### Running in development mode with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Running in production:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🌍 Global Data Sources

The module integrates data from:

- **Europe/ESA**: Planck Legacy Archive
- **USA/International**: DESI DR1/DR2, Pantheon+
- **Japan**: Subaru HSC PDR3
- **China**: LAMOST DR10
- **Russia/Germany**: eROSITA eRASS1
- **India**: AstroSat (ISRO), GMRT (NCRA/TIFR)
- **International**: JWST COSMOS-Web

## ⚙️ Configuration

Default simulation parameters can be modified in the `SimulationParams` class in `main.py`:

```python
class SimulationParams(BaseModel):
    H0_kms_mpc: float = 67.4      # Hubble constant
    omega_b: float = 0.0493        # Baryon density
    omega_c: float = 0.264         # CDM density
    omega_r: float = 0.000092      # Radiation density
    lambda_phi: float = 0.1        # Scalar field parameter
    V0: float = 0.757              # Potential normalization
    xi: float = 0.1666667          # Non-minimal coupling
    omega_k: float = -0.07         # Curvature density
    # ... more parameters
```

## 📝 License

This software is released for academic transparency and non-commercial scientific research. See the license terms in `dut_module_engine.py` for details.

**ExtractoDAO Labs** | CNPJ: 48.839.397/0001-36
Contact: contato@extractodao.com

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Make sure dut_module_engine.py is in the same directory as main.py
# and all dependencies are installed
pip install -r requirements.txt
```

### Issue: Port already in use

```bash
# Use a different port
uvicorn main:app --port 8080
```

### Issue: Simulation takes too long

The first run may take 30-60 seconds as it downloads Pantheon+ data. Subsequent runs use cached data and complete in ~5-10 seconds.

## 🤝 Contributing

For questions, issues, or contributions, please contact ExtractoDAO Labs.

---

**Version**: 1.0.0
**Last Updated**: December 2024
