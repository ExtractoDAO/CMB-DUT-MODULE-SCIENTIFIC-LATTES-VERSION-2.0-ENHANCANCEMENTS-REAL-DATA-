#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Web Service for DUT-CMB Module
Serves simulation results with interactive visualization
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import numpy as np
import io
import base64
import json
from datetime import datetime

# Import the DUT module
import dut_module_engine as dut

app = FastAPI(
    title="DUT-CMB Cosmological Modeling API",
    description="Dead Universe Theory - CMB Module Scientific Computing Interface",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for simulation results
_simulation_cache: Optional[Dict[str, Any]] = None


class SimulationParams(BaseModel):
    H0_kms_mpc: float = 67.4
    omega_b: float = 0.0493
    omega_c: float = 0.264
    omega_r: float = 0.000092
    omega_nu: float = 0.0
    n_eff: float = 3.044
    lambda_phi: float = 0.1
    V0: float = 0.757
    xi: float = 0.1666667
    omega_k: float = -0.07
    phi_ini: float = 0.0
    dphi_dN_ini: float = 0.0


class SimulationResult(BaseModel):
    status: str
    timestamp: str
    parameters: Dict[str, float]
    cmb_observables: Dict[str, float]
    unit_check: bool
    computation_time: float
    data_points: Dict[str, List[float]]


def run_dut_simulation(params: SimulationParams) -> Dict[str, Any]:
    """Execute DUT simulation with given parameters"""
    import time
    start_time = time.time()

    # Create DUT parameters
    dut_params = dut.DUTParameters(
        H0_kms_mpc=params.H0_kms_mpc,
        omega_b=params.omega_b,
        omega_c=params.omega_c,
        omega_r=params.omega_r,
        omega_nu=params.omega_nu,
        n_eff=params.n_eff,
        lambda_phi=params.lambda_phi,
        V0=params.V0,
        xi=params.xi,
        omega_k=params.omega_k,
        phi_ini=params.phi_ini,
        dphi_dN_ini=params.dphi_dN_ini,
    )

    # Generate background tables
    bg_data = dut.BackgroundSolver.generate_background_tables(
        dut_params, a_ini=1e-6, a_final=1.0, N_points=900
    )

    z_tab = (1.0 / np.maximum(bg_data["a"], 1e-60)) - 1.0
    H_tab = bg_data["H"]

    # Run perturbations
    driver = dut.PerturbationsDriver(bg_data)
    pert_res = driver.solve_perturbations(dut_params, k_mode=0.01, a_ini=1e-4)

    z_eval = np.linspace(0.0, 2.5, 60)
    a_eval = 1.0 / (1.0 + z_eval)
    fs8_vals = dut.GrowthFactorModule.compute_fsig8(
        a_eval, pert_res, dut_params, sigma8_0=0.81
    )

    # CMB observables
    cfg = dut.CMBConfig(
        omega_b=dut_params.omega_b,
        omega_c=dut_params.omega_c,
        omega_r=dut_params.omega_r,
        omega_nu=dut_params.omega_nu,
        tau_target=1.0
    )
    pri = dut.CMBModule.default_planck_lA_prior(sigma=0.3)
    H_of_a = bg_data["H_of_a_E_interp"]
    lnlike, obs = dut.CMBModule.lnlike_cmb(H_of_a, dut_params.H0_kms_mpc, cfg, pri)

    # Unit check
    unit_check = dut._quick_unit_check_lA(obs, target=301.6, tol=3.0)

    computation_time = time.time() - start_time

    # Prepare output data
    mask = (z_tab >= 0.0) & (z_tab <= 10.0)

    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parameters": params.dict(),
        "cmb_observables": {
            "lA": float(obs["lA"]),
            "z_th": float(obs["z_th"]),
            "tau_th": float(obs["tau_th"]),
            "D_M_th": float(obs["D_M_th"]),
            "r_s_th": float(obs["r_s_th"]),
            "lnlike": float(lnlike)
        },
        "unit_check": bool(unit_check),
        "computation_time": float(computation_time),
        "data_points": {
            "z_hubble": z_tab[mask].tolist(),
            "H_z": H_tab[mask].tolist(),
            "z_growth": z_eval.tolist(),
            "fsig8": fs8_vals.tolist()
        }
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main HTML interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DUT-CMB Cosmological Modeling</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            padding: 30px 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .subtitle {
            font-size: 1.1em;
            color: #b0b0b0;
            margin-top: 10px;
        }

        .control-panel {
            background: rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }

        .button-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }

        button {
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .status-panel {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: none;
        }

        .status-panel.active {
            display: block;
        }

        .loading {
            text-align: center;
            padding: 40px;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top: 4px solid #667eea;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }

        .result-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: #b0b0b0;
        }

        .metric-value {
            color: #fff;
            font-weight: 600;
        }

        .badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }

        .badge-success {
            background: rgba(76, 175, 80, 0.3);
            color: #4caf50;
        }

        .badge-warning {
            background: rgba(255, 152, 0, 0.3);
            color: #ff9800;
        }

        .plot-container {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
        }

        .plot-container h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }

        footer {
            text-align: center;
            padding: 30px;
            color: #808080;
            font-size: 0.9em;
        }

        .citation {
            background: rgba(255, 255, 255, 0.03);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌌 DUT-CMB Cosmological Modeling</h1>
            <p class="subtitle">Dead Universe Theory - Scientific Lattes Version</p>
            <p class="subtitle">Enhanced with Global Observational Data</p>
        </header>

        <div class="control-panel">
            <div class="button-group">
                <button class="btn-primary" onclick="runSimulation()">
                    🚀 Run Simulation
                </button>
                <button class="btn-secondary" onclick="downloadResults()">
                    💾 Download Results (JSON)
                </button>
                <button class="btn-secondary" onclick="viewDataSources()">
                    🌍 Global Data Sources
                </button>
            </div>
        </div>

        <div id="loading" class="status-panel">
            <div class="loading">
                <div class="spinner"></div>
                <p>Running cosmological simulation...</p>
                <p style="color: #808080; margin-top: 10px;">Computing background evolution, perturbations, and CMB observables</p>
            </div>
        </div>

        <div id="results" class="status-panel">
            <div class="results-grid">
                <div class="result-card">
                    <h3>CMB Observables</h3>
                    <div id="cmb-metrics"></div>
                </div>

                <div class="result-card">
                    <h3>Simulation Status</h3>
                    <div id="status-metrics"></div>
                </div>

                <div class="result-card">
                    <h3>Model Parameters</h3>
                    <div id="param-metrics"></div>
                </div>
            </div>

            <div class="plot-container">
                <h3>Hubble Parameter Evolution H(z)</h3>
                <div id="hubble-plot"></div>
            </div>

            <div class="plot-container">
                <h3>Growth Rate fσ₈(z)</h3>
                <div id="growth-plot"></div>
            </div>
        </div>

        <footer>
            <div class="citation">
                <strong>Citation Required:</strong><br>
                Almeida, J. (2025). Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
                Hubble and Growth Tensions Simultaneously: Δχ² = –211.6 with Identical Datasets.
                Zenodo. <a href="https://doi.org/10.5281/zenodo.17752029" style="color: #667eea;">
                https://doi.org/10.5281/zenodo.17752029</a>
            </div>
            <p style="margin-top: 20px;">ExtractoDAO Labs | CNPJ: 48.839.397/0001-36</p>
            <p>Contact: contato@extractodao.com</p>
        </footer>
    </div>

    <script>
        let currentResults = null;

        async function runSimulation() {
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').classList.remove('active');

            try {
                const response = await fetch('/api/simulate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });

                if (!response.ok) {
                    throw new Error('Simulation failed');
                }

                currentResults = await response.json();
                displayResults(currentResults);

            } catch (error) {
                alert('Error running simulation: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        }

        function displayResults(data) {
            // CMB Observables
            const cmbHTML = `
                <div class="metric">
                    <span class="metric-label">lA (acoustic scale)</span>
                    <span class="metric-value">${data.cmb_observables.lA.toFixed(4)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">z_th (thermalization)</span>
                    <span class="metric-value">${data.cmb_observables.z_th.toFixed(2)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ln(likelihood)</span>
                    <span class="metric-value">${data.cmb_observables.lnlike.toFixed(4)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">r_s (sound horizon)</span>
                    <span class="metric-value">${data.cmb_observables.r_s_th.toFixed(2)} Mpc</span>
                </div>
            `;
            document.getElementById('cmb-metrics').innerHTML = cmbHTML;

            // Status
            const statusHTML = `
                <div class="metric">
                    <span class="metric-label">Unit Check (lA ≈ 301.6)</span>
                    <span class="metric-value">
                        <span class="badge ${data.unit_check ? 'badge-success' : 'badge-warning'}">
                            ${data.unit_check ? 'PASS ✓' : 'FAIL ✗'}
                        </span>
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Computation Time</span>
                    <span class="metric-value">${data.computation_time.toFixed(2)}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Timestamp</span>
                    <span class="metric-value">${new Date(data.timestamp).toLocaleString()}</span>
                </div>
            `;
            document.getElementById('status-metrics').innerHTML = statusHTML;

            // Parameters
            const paramHTML = `
                <div class="metric">
                    <span class="metric-label">H₀</span>
                    <span class="metric-value">${data.parameters.H0_kms_mpc.toFixed(2)} km/s/Mpc</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Ω_b</span>
                    <span class="metric-value">${data.parameters.omega_b.toFixed(4)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Ω_c</span>
                    <span class="metric-value">${data.parameters.omega_c.toFixed(4)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">λ_φ</span>
                    <span class="metric-value">${data.parameters.lambda_phi.toFixed(4)}</span>
                </div>
            `;
            document.getElementById('param-metrics').innerHTML = paramHTML;

            // Plots
            plotHubble(data.data_points);
            plotGrowth(data.data_points);

            document.getElementById('results').classList.add('active');
        }

        function plotHubble(data) {
            const trace = {
                x: data.z_hubble,
                y: data.H_z,
                mode: 'lines',
                name: 'H(z)',
                line: {
                    color: '#667eea',
                    width: 3
                }
            };

            const layout = {
                xaxis: {
                    title: 'Redshift z',
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    color: '#e0e0e0'
                },
                yaxis: {
                    title: 'H(z) [km/s/Mpc]',
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    color: '#e0e0e0'
                },
                plot_bgcolor: 'rgba(0, 0, 0, 0)',
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                font: { color: '#e0e0e0' }
            };

            Plotly.newPlot('hubble-plot', [trace], layout, {responsive: true});
        }

        function plotGrowth(data) {
            const trace = {
                x: data.z_growth,
                y: data.fsig8,
                mode: 'lines',
                name: 'fσ₈(z)',
                line: {
                    color: '#764ba2',
                    width: 3
                }
            };

            const layout = {
                xaxis: {
                    title: 'Redshift z',
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    color: '#e0e0e0'
                },
                yaxis: {
                    title: 'fσ₈(z)',
                    gridcolor: 'rgba(255, 255, 255, 0.1)',
                    color: '#e0e0e0'
                },
                plot_bgcolor: 'rgba(0, 0, 0, 0)',
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                font: { color: '#e0e0e0' }
            };

            Plotly.newPlot('growth-plot', [trace], layout, {responsive: true});
        }

        function downloadResults() {
            if (!currentResults) {
                alert('No results to download. Please run a simulation first.');
                return;
            }

            const dataStr = JSON.stringify(currentResults, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `dut-cmb-results-${new Date().toISOString()}.json`;
            link.click();
        }

        function viewDataSources() {
            window.open('/api/data-sources', '_blank');
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate(params: Optional[SimulationParams] = None):
    """Run DUT simulation with optional custom parameters"""
    try:
        if params is None:
            params = SimulationParams()

        result = run_dut_simulation(params)

        # Cache the result
        global _simulation_cache
        _simulation_cache = result

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/api/data-sources", response_class=HTMLResponse)
async def data_sources():
    """Display global data sources information"""
    sources = dut.GlobalCosmoData().list_all_sources()

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Global Data Sources</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            h1 { color: #333; }
            .source {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .source h3 { color: #667eea; margin: 0 0 10px 0; }
            a { color: #667eea; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>🌍 Global Cosmological Data Sources</h1>
        <p>The DUT-CMB module integrates data from major international observatories and surveys:</p>
    """

    for name, url in sources.items():
        html += f"""
        <div class="source">
            <h3>{name}</h3>
            <a href="{url}" target="_blank">{url}</a>
        </div>
        """

    html += """
    </body>
    </html>
    """

    return HTMLResponse(content=html)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/cache")
async def get_cache():
    """Retrieve cached simulation results"""
    if _simulation_cache is None:
        raise HTTPException(status_code=404, detail="No cached results available")
    return _simulation_cache


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)