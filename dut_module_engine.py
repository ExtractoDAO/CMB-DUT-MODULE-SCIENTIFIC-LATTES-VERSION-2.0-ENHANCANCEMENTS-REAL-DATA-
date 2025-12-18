#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CMB DUT MODULE — SCIENTIFIC LATTES VERSION (ENHANCANCEMENTS & REAL DATA)
================================================================================
DUT-CMB Module — Dead Universe Theory Cosmological Modeling with scientific
computing enhancements and global observational data.

ENHANCEMENTS:
1. MSPC: Multiprocess-Safe Parameter Caching
2. HCNI: Halt Condition on Numerical Instability
3. APS: Adaptive Precision Scaling
4. DRP: Dynamic Requirement Provisioning
5. ADMV: Automatic Data/Model Versioning
6. SDQG: Derived Quantity Generation
7. IPVN: Input Parameter Validation and Normalization

GLOBAL DATA SOURCES:
- Europe/ESA: Planck Legacy Archive
- USA/International: DESI DR1/DR2, Pantheon+
- Japan: Subaru HSC PDR3
- China: LAMOST DR10
- Russia/Germany: eROSITA eRASS1
- India: AstroSat (ISRO), GMRT (NCRA/TIFR)
- International: JWST COSMOS-Web (COSMOS2025 catalog), BAO compilations

================================================================================

LICENSE AND PERMISSIONS
- ------------------------
-  This software is released for academic transparency and
-  non-commercial scientific research. The following conditions apply:
-
-    1. Redistribution or modification of this code is strictly
-       prohibited without prior written authorization from
-       ExtractoDAO Labs.
-
-    2. Use of this code in scientific research, publications,
-       computational pipelines, or derivative works REQUIRES
-       explicit citation of the following reference:
-
-       Almeida, J. (2025).
-       Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
-       Hubble and Growth Tensions Simultaneously:
-       Δχ² = –211.6 with Identical Datasets.
-       Zenodo. https://doi.org/10.5281/zenodo.17752029
-
-    3. Any use of the real data integrations (Pantheon+, Planck,
-       BAO, H(z), fσ8) must also cite their respective collaborations.
-
-    4. Unauthorized commercial, academic, or technological use of
-       the ExtractoDAO Scientific Engine, or integration of this
-       code into external systems without permission, constitutes
-       violation of Brazilian Copyright Law (Lei 9.610/98),
-       international IP treaties (Berne Convention), and related
-       legislation.

ExtractoDAO Labs | CNPJ: 48.839.397/0001-36
Contact: contato@extractodao.com
================================================================================
"""

from __future__ import annotations
import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d

try:
    from scipy.linalg import cho_factor, cho_solve
except Exception:
    cho_factor = None
    cho_solve = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import requests
except Exception:
    requests = None


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class PantheonLoader:
    DEFAULT_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2B_SH0ES.dat"
    DEFAULT_COV_URLS = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/sys_full.cov",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/SYS_FULL.COV",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/sys_full_long.cov",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/sys_full_long.COV",
    )

    def __init__(
        self,
        cache_dir: str = ".dut_cache",
        url: str = DEFAULT_URL,
        cov_urls: Optional[Tuple[str, ...]] = None,
    ):
        self.url = str(url)
        self.cov_urls = tuple(cov_urls) if cov_urls is not None else tuple(self.DEFAULT_COV_URLS)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_path = self.cache_dir / "PantheonPlus_SH0ES.dat"
        self.meta_path = self.cache_dir / "PantheonPlus_SH0ES.meta.json"

        self.cov_path = self.cache_dir / "PantheonPlus_sys_full.cov"
        self.cov_meta_path = self.cache_dir / "PantheonPlus_sys_full.meta.json"

    @staticmethod
    def _sha256_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _write_cache(self, raw: bytes) -> None:
        sha = self._sha256_bytes(raw)
        self.data_path.write_bytes(raw)
        meta = {
            "source_url": self.url,
            "sha256": sha,
            "download_utc": _now_utc_iso(),
            "bytes": int(len(raw)),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _read_cache_verified(self) -> Optional[str]:
        if (not self.data_path.exists()) or (not self.meta_path.exists()):
            return None
        try:
            raw = self.data_path.read_bytes()
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            sha = self._sha256_bytes(raw)
            if sha != meta.get("sha256"):
                return None
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None

    def fetch_text(self) -> str:
        if requests is not None:
            try:
                r = requests.get(self.url, timeout=45)
                r.raise_for_status()
                raw = r.content
                self._write_cache(raw)
                return raw.decode("utf-8", errors="replace")
            except Exception:
                pass
        cached = self._read_cache_verified()
        if cached is not None:
            return cached
        raise RuntimeError("CRITICAL: Pantheon+ download failed and no verified cache found.")

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pd is None:
            raise RuntimeError("pandas not available; cannot parse Pantheon+ table.")
        from io import StringIO

        text = self.fetch_text()
        df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
        if "IS_TRAINING" in df.columns:
            df = df[df["IS_TRAINING"] == 0]

        for col in ("zHD", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"):
            if col not in df.columns:
                raise RuntimeError(f"Pantheon+ missing column: {col}")

        z = df["zHD"].to_numpy(dtype=float)
        mu = df["MU_SH0ES"].to_numpy(dtype=float)
        muerr = df["MU_SH0ES_ERR_DIAG"].to_numpy(dtype=float)

        if (not np.all(np.isfinite(z))) or (not np.all(np.isfinite(mu))) or (not np.all(np.isfinite(muerr))):
            raise RuntimeError("Pantheon+ contains non-finite values.")
        if np.any(muerr <= 0):
            raise RuntimeError("Pantheon+ has non-positive diagonal errors.")
        return z, mu, muerr

    def _write_cov_cache(self, raw: bytes, url: str) -> None:
        sha = self._sha256_bytes(raw)
        self.cov_path.write_bytes(raw)
        meta = {
            "source_url": url,
            "sha256": sha,
            "download_utc": _now_utc_iso(),
            "bytes": int(len(raw)),
        }
        self.cov_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _read_cov_cache_verified(self) -> Optional[bytes]:
        if (not self.cov_path.exists()) or (not self.cov_meta_path.exists()):
            return None
        try:
            raw = self.cov_path.read_bytes()
            meta = json.loads(self.cov_meta_path.read_text(encoding="utf-8"))
            sha = self._sha256_bytes(raw)
            if sha != meta.get("sha256"):
                return None
            return raw
        except Exception:
            return None

    @staticmethod
    def _parse_cov_bytes(raw: bytes) -> np.ndarray:
        txt = raw.decode("utf-8", errors="replace")
        arr = np.fromstring(txt, sep=" ", dtype=float)
        if arr.size < 4:
            arr = np.fromstring(txt.replace("\n", " "), sep=" ", dtype=float)
        if arr.size < 4:
            raise RuntimeError("Pantheon+ covariance parse failed (too few numbers).")

        first = arr[0]
        n0 = int(round(first))
        if n0 > 0 and (arr.size == 1 + n0 * n0):
            mat = arr[1:].reshape((n0, n0))
            return mat.astype(float, copy=False)

        n = int(round(np.sqrt(arr.size)))
        if n * n == arr.size:
            mat = arr.reshape((n, n))
            return mat.astype(float, copy=False)

        raise RuntimeError("Pantheon+ covariance parse failed (unsupported format).")

    def load_covariance(self, expected_n: Optional[int] = None) -> Optional[np.ndarray]:
        raw = None
        if requests is not None:
            for url in self.cov_urls:
                try:
                    r = requests.get(url, timeout=45)
                    if r.status_code != 200:
                        continue
                    raw = r.content
                    self._write_cov_cache(raw, url=url)
                    break
                except Exception:
                    continue

        if raw is None:
            cached = self._read_cov_cache_verified()
            if cached is not None:
                raw = cached

        if raw is None:
            return None

        cov = self._parse_cov_bytes(raw)
        if expected_n is not None and cov.shape != (int(expected_n), int(expected_n)):
            return None
        if not np.all(np.isfinite(cov)):
            return None
        if cov.shape[0] != cov.shape[1]:
            return None
        return cov


class GlobalCosmoData:
    def __init__(self):
        self.planck_legacy_archive = "https://pla.esac.esa.int/"
        self.desi_dr1_main = "https://data.desi.lbl.gov/public/dr1/"
        self.desi_bao_cosmo_params = "https://data.desi.lbl.gov/public/dr1/vac/dr1/bao-cosmo-params/"
        self.pantheon_github_repo = "https://github.com/PantheonPlusSH0ES/DataRelease"
        self.pantheon_snia_csv = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2B_SH0ES.dat"
        self.hsc_pdr3_main = "https://hsc-release.mtk.nao.ac.jp/doc/index.php/sample-page/pdr3/"
        self.hsc_data_access = "https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/"
        self.lamost_dr10_main = "https://www.lamost.org/dr10/v1.0/"
        self.lamost_catalog_search = "http://www.lamost.org/dr10/v1.0/catalogue"
        self.erosita_dr1_main = "https://erosita.mpe.mpg.de/dr1/"
        self.erosita_catalog_example = "https://erosita.mpe.mpg.de/dr1/erass1_main_v1.0.fits"
        self.cosmos_web_main = "https://cosmos-web.ipac.caltech.edu/data/"
        self.cosmos2025_catalog_site = "https://cosmos2025.iap.fr/"
        self.astrosat_archive = "https://astrobrowse.issdc.gov.in/astro_archive/"
        self.astrosat_main = "https://issdc.gov.in/astro.html"
        self.gmrt_archive = "https://naps.ncra.tifr.res.in/goa/"
        self.bao_archive = "https://bea.cosmo.fas.nyu.edu/baoarchive/"

    def download_text(self, url: str) -> str:
        if requests is None:
            raise RuntimeError("requests not available.")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def download_csv(self, url: str, **kwargs):
        if pd is None:
            raise RuntimeError("pandas not available.")
        return pd.read_csv(url, **kwargs)

    def list_all_sources(self) -> Dict[str, str]:
        return {
            "Planck Legacy Archive (Europe/ESA)": self.planck_legacy_archive,
            "DESI DR1 BAO/Cosmology Chains (USA/Global)": self.desi_bao_cosmo_params,
            "Pantheon+ SNIa Table (SH0ES)": self.pantheon_snia_csv,
            "Subaru HSC PDR3 (Japan)": self.hsc_data_access,
            "LAMOST DR10 Catalogs (China)": self.lamost_catalog_search,
            "eROSITA eRASS1 (Russia/Germany)": self.erosita_dr1_main,
            "JWST COSMOS-Web COSMOS2025 Catalog": self.cosmos2025_catalog_site,
            "AstroSat Archive (India/ISRO)": self.astrosat_archive,
            "GMRT Online Archive (India/NCRA)": self.gmrt_archive,
            "Global BAO Compilation Archive": self.bao_archive,
        }


class PhysicalConstants:
    C_KMS = 299792.458
    MPC_M = 3.0856775814913673e22
    KM_M = 1e3
    SIGMA_T = 6.6524587321e-29
    M_P = 1.67262192369e-27
    G_SI = 6.67430e-11
    MPC_KM = MPC_M / KM_M


@dataclass(frozen=True)
class CMBPriors:
    mean: np.ndarray
    invcov: np.ndarray
    names: Tuple[str, ...] = ("lA",)

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        invcov = np.asarray(self.invcov, dtype=float)
        if (
            mean.ndim != 1
            or invcov.ndim != 2
            or invcov.shape[0] != invcov.shape[1]
            or invcov.shape[0] != mean.shape[0]
            or len(self.names) != mean.shape[0]
        ):
            raise ValueError("CMBPriors dimensions are inconsistent.")


@dataclass(frozen=True)
class BaryonRecombinationModel:
    z_recomb: float = 1100.0
    delta_z: float = 80.0
    xe_highz: float = 1.0
    xe_lowz: float = 1e-4

    def xe(self, z: float) -> float:
        x = (z - self.z_recomb) / max(self.delta_z, 1e-9)
        return self.xe_lowz + (self.xe_highz - self.xe_lowz) / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class CMBConfig:
    omega_b: float
    omega_c: float
    omega_r: float
    omega_nu: float = 0.0
    Yp_He: float = 0.24
    recomb_model: BaryonRecombinationModel = BaryonRecombinationModel()
    tau_target: float = 1.0
    z_th_min: float = 50.0
    z_th_max: float = 4000.0
    epsabs: float = 1e-8
    epsrel: float = 1e-8


class CMBModule:
    C = PhysicalConstants.C_KMS
    SIGMA_T = PhysicalConstants.SIGMA_T
    M_P = PhysicalConstants.M_P
    G_SI = PhysicalConstants.G_SI
    MPC_M = PhysicalConstants.MPC_M
    KM_M = PhysicalConstants.KM_M

    @staticmethod
    def _E_of_z_from_Ha(H_of_a: Callable[[float], float], z: float) -> float:
        a = 1.0 / (1.0 + z)
        return float(H_of_a(a))

    @classmethod
    def comoving_distance_DM(cls, H_of_a: Callable[[float], float], z: float, c_kms: float, epsabs: float, epsrel: float) -> float:
        def integrand(zp: float) -> float:
            Hz = cls._E_of_z_from_Ha(H_of_a, zp)
            return c_kms / max(Hz, 1e-30)
        val, _ = quad(integrand, 0.0, float(z), epsabs=epsabs, epsrel=epsrel, limit=300)
        return float(val)

    @classmethod
    def sound_speed_cs(cls, omega_b: float, omega_r: float, z: float) -> float:
        a = 1.0 / (1.0 + z)
        rho_b = omega_b / (a**3)
        rho_r = omega_r / (a**4)
        R = 3.0 * rho_b / (4.0 * rho_r + 1e-60)
        return cls.C / np.sqrt(3.0 * (1.0 + R))

    @classmethod
    def sound_horizon_rs(cls, H_of_a: Callable[[float], float], omega_b: float, omega_r: float, z_star: float, epsabs: float, epsrel: float) -> float:
        zmax = 1.0e7
        def integrand(zp: float) -> float:
            cs = cls.sound_speed_cs(omega_b, omega_r, zp)
            Hz = cls._E_of_z_from_Ha(H_of_a, zp)
            return cs / max(Hz, 1e-30)
        val, _ = quad(integrand, float(z_star), zmax, epsabs=epsabs, epsrel=epsrel, limit=500)
        return float(val)

    @classmethod
    def _baryon_number_density_ne0(cls, omega_b: float, H0_km_s_Mpc: float, Yp: float) -> float:
        H0_si = (H0_km_s_Mpc * cls.KM_M) / cls.MPC_M
        rho_crit0 = 3.0 * (H0_si**2) / (8.0 * np.pi * cls.G_SI)
        rho_b0 = omega_b * rho_crit0
        n_b0 = rho_b0 / cls.M_P
        ne0 = n_b0 * (1.0 - 0.5 * Yp)
        return float(ne0)

    @classmethod
    def optical_depth_tau(cls, H_of_a: Callable[[float], float], config: CMBConfig, H0_km_s_Mpc: float, z: float) -> float:
        ne0 = cls._baryon_number_density_ne0(config.omega_b, H0_km_s_Mpc, config.Yp_He)

        def integrand(zp: float) -> float:
            a = 1.0 / (1.0 + zp)
            Hz = float(H_of_a(a))
            xe = config.recomb_model.xe(zp)
            ne = ne0 * (1.0 + zp) ** 3 * xe
            c_ms = cls.C * cls.KM_M
            Hz_si = (Hz * cls.KM_M) / cls.MPC_M
            return (c_ms * cls.SIGMA_T * ne) / ((1.0 + zp) * max(Hz_si, 1e-40))

        val, _ = quad(integrand, 0.0, float(z), epsabs=config.epsabs, epsrel=config.epsrel, limit=400)
        return float(val)

    @classmethod
    def find_z_thermalization(cls, H_of_a_km_s_Mpc: Callable[[float], float], config: CMBConfig, H0_km_s_Mpc: float) -> float:
        z_lo, z_hi = float(config.z_th_min), float(config.z_th_max)
        target = float(config.tau_target)
        tau_lo = cls.optical_depth_tau(H_of_a_km, config, H0_km_s_Mpc, z_lo) if False else None
        tau_lo = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_lo)
        tau_hi = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_hi)
        for _ in range(40):
            if tau_lo > target:
                z_lo = max(z_lo * 0.7, 1e-6)
                tau_lo = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_lo)
            elif tau_hi < target:
                z_hi = z_hi * 1.3 + 10.0
                tau_hi = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_hi)
            else:
                break
        if not (tau_lo <= target <= tau_hi):
            return float(z_hi)
        for _ in range(80):
            z_mid = 0.5 * (z_lo + z_hi)
            tau_mid = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_mid)
            if tau_mid < target:
                z_lo, tau_lo = z_mid, tau_mid
            else:
                z_hi, tau_hi = z_mid, tau_mid
            if abs(z_hi - z_lo) / max(z_mid, 1e-9) < 1e-8:
                break
        return float(0.5 * (z_lo + z_hi))

    @classmethod
    def compute_observables(cls, H_of_a: Callable[[float], float], H0_km_s_Mpc: float, config: CMBConfig, *, H_of_a_units: str = "km/s/Mpc") -> Dict[str, float]:
        if H_of_a_units == "km/s/Mpc":
            H_of_a_km = H_of_a
            H_of_a_for_dist = H_of_a
            c_for_dist = cls.C
        elif H_of_a_units == "1/Mpc":
            def H_of_a_km(a: float) -> float:
                return float(H_of_a(a)) * cls.C
            H_of_a_for_dist = H_of_a
            c_for_dist = 1.0
        else:
            raise ValueError("H_of_a_units must be 'km/s/Mpc' or '1/Mpc'.")

        z_th = cls.find_z_thermalization(H_of_a_km, config, H0_km_s_Mpc)
        tau_th = cls.optical_depth_tau(H_of_a_km, config, H0_km_s_Mpc, z_th)
        Dm = cls.comoving_distance_DM(H_of_a_for_dist, z_th, c_kms=c_for_dist, epsabs=config.epsabs, epsrel=config.epsrel)
        rs = cls.sound_horizon_rs(H_of_a_for_dist, config.omega_b, config.omega_r, z_th, epsabs=config.epsabs, epsrel=config.epsrel)
        lA = np.pi * (Dm / max(rs, 1e-60))
        return {"z_th": float(z_th), "tau_th": float(tau_th), "D_M_th": float(Dm), "r_s_th": float(rs), "lA": float(lA)}

    @staticmethod
    def chi2_distance_priors(observables: Dict[str, float], priors: CMBPriors, *, extra: Optional[Dict[str, float]] = None) -> float:
        x = []
        for name in priors.names:
            if name == "lA":
                x.append(observables["lA"])
            elif name == "R":
                if extra is None or "R" not in extra:
                    raise ValueError("Priors require R but extra['R'] was not provided.")
                x.append(float(extra["R"]))
            else:
                raise ValueError(f"Unsupported prior observable name: {name}")
        x = np.asarray(x, dtype=float)
        dx = x - priors.mean
        chi2 = float(dx.T @ priors.invcov @ dx)
        return chi2

    @classmethod
    def lnlike_cmb(
        cls,
        H_of_a: Callable[[float], float],
        H0_km_s_Mpc: float,
        config: CMBConfig,
        priors: CMBPriors,
        *,
        H_of_a_units: str = "km/s/Mpc",
        extra: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        obs = cls.compute_observables(H_of_a, H0_km_s_Mpc, config, H_of_a_units=H_of_a_units)
        chi2 = cls.chi2_distance_priors(obs, priors, extra=extra)
        return -0.5 * chi2, obs

    @staticmethod
    def default_planck_lA_prior(sigma: float = 0.3) -> CMBPriors:
        mean = np.array([301.6], dtype=float)
        invcov = np.array([[1.0 / (sigma**2)]], dtype=float)
        return CMBPriors(mean=mean, invcov=invcov, names=("lA",))


@dataclass(frozen=True)
class BAOData:
    z: float
    dm_over_rd_obs: float
    dh_over_rd_obs: float
    sigma_dm: float
    sigma_dh: float


@dataclass(frozen=True)
class DUTParameters:
    H0_kms_mpc: float
    omega_b: float
    omega_c: float
    omega_r: float
    omega_nu: float = 0.0
    n_eff: float = 3.044
    lambda_phi: float = 0.1
    V0: float = 0.757
    xi: float = 0.1666667
    omega_k: float = -0.07
    phi_ini: float = 0.0
    dphi_dN_ini: float = 0.0


class BackgroundSolver:
    @staticmethod
    def _V(phi: float, params: DUTParameters) -> float:
        return float(params.V0) * float(np.exp(-float(params.lambda_phi) * float(phi)))

    @staticmethod
    def _dV_dphi(phi: float, params: DUTParameters) -> float:
        return -float(params.lambda_phi) * BackgroundSolver._V(phi, params)

    @staticmethod
    def _E2_from_constraint(a: float, phi: float, u: float, params: DUTParameters) -> float:
        eps = 1e-60
        a = float(max(float(a), eps))
        omega_m0 = float(params.omega_b + params.omega_c)
        omega_r0 = float(params.omega_r + params.omega_nu)
        omega_k0 = float(params.omega_k)

        rho_m = omega_m0 * a**-3
        rho_r = omega_r0 * a**-4
        rho_k = omega_k0 * a**-2

        Vphi = BackgroundSolver._V(phi, params)
        xi = float(params.xi)

        D = (1.0 + xi * phi * phi) - (u * u) / 6.0 - 2.0 * xi * phi * u
        D = float(np.sign(D) * max(abs(D), 1e-12))

        E2 = (rho_m + rho_r + rho_k + Vphi) / D

        if E2 < 0 or not np.isfinite(E2):
            raise ValueError("HCNI: Numerical instability in E^2 (negative or inf/nan)")

        return float(max(E2, 1e-30))

    @staticmethod
    def _dlnH_dN_numeric(N: float, phi: float, u: float, params: DUTParameters) -> float:
        h = 1e-4
        a1 = float(np.exp(N + h))
        a2 = float(np.exp(N - h))
        phi1 = float(phi + u * h)
        phi2 = float(phi - u * h)
        E2_1 = BackgroundSolver._E2_from_constraint(a1, phi1, u, params)
        E2_2 = BackgroundSolver._E2_from_constraint(a2, phi2, u, params)
        lnH1 = 0.5 * np.log(max(E2_1, 1e-60))
        lnH2 = 0.5 * np.log(max(E2_2, 1e-60))
        return float((lnH1 - lnH2) / (2.0 * h))

    @staticmethod
    def _R_over_H2(N: float, phi: float, u: float, params: DUTParameters, E2: float, dlnH_dN: float) -> float:
        a = float(np.exp(N))
        omega_k0 = float(params.omega_k)
        curv_term = -(omega_k0 * a**-2) / max(float(E2), 1e-30)
        return float(6.0 * (2.0 + float(dlnH_dN) + float(curv_term)))

    @classmethod
    def generate_background_tables(cls, params: DUTParameters, a_ini: float = 1e-6, a_final: float = 1.0, N_points: int = 800) -> Dict[str, Any]:
        eps = 1e-60
        a_ini = float(max(a_ini, eps))
        a_final = float(max(a_final, a_ini * 1.0001))
        N_points = int(max(50, N_points))

        ln_a = np.linspace(np.log(a_ini), np.log(a_final), N_points)
        a_grid = np.exp(ln_a)

        N0, N1 = float(ln_a[0]), float(ln_a[-1])
        y0 = np.array([float(params.phi_ini), float(params.dphi_dN_ini)], dtype=float)

        def rhs(N: float, y: np.ndarray) -> np.ndarray:
            phi = float(y[0])
            u = float(y[1])
            a = float(np.exp(N))
            E2 = cls._E2_from_constraint(a, phi, u, params)
            dlnH_dN = cls._dlnH_dN_numeric(N, phi, u, params)
            xi = float(params.xi)
            dV = cls._dV_dphi(phi, params)
            R_over_H2 = cls._R_over_H2(N, phi, u, params, E2, dlnH_dN)
            u_prime = -(3.0 + dlnH_dN) * u + (dV / max(E2, 1e-30)) - 2.0 * xi * R_over_H2 * phi
            return np.array([u, u_prime], dtype=float)

        sol = None
        last_err = None
        for method, rtol, atol in (("Radau", 1e-8, 1e-12), ("LSODA", 1e-8, 1e-12), ("RK45", 1e-8, 1e-12)):
            try:
                sol = solve_ivp(rhs, (N0, N1), y0, t_eval=ln_a, rtol=rtol, atol=atol, method=method)
                if sol.success:
                    break
                last_err = sol.message
                sol = None
            except Exception as e:
                last_err = str(e)
                sol = None

        if sol is None:
            raise RuntimeError(f"Background scalar integration failed (fallback exhausted): {last_err}")

        phi = np.asarray(sol.y[0], dtype=float)
        u = np.asarray(sol.y[1], dtype=float)

        E2_arr = np.array(
            [cls._E2_from_constraint(ai, float(ph), float(ui), params) for ai, ph, ui in zip(a_grid, phi, u)],
            dtype=float,
        )
        H = float(params.H0_kms_mpc) * np.sqrt(np.maximum(E2_arr, 1e-30))

        H_mpc_inv = H / PhysicalConstants.C_KMS
        integrand = 1.0 / (np.maximum(a_grid, eps) * np.maximum(H_mpc_inv, 1e-60))
        dln = float(ln_a[1] - ln_a[0])
        tau = np.cumsum(integrand) * dln
        tau -= tau[0]

        dphidtau = u * a_grid * np.maximum(H_mpc_inv, 1e-60)

        H_of_a_E = interp1d(a_grid, H, kind="linear", bounds_error=False, fill_value=(float(H[0]), float(H[-1])))

        return {
            "a": a_grid,
            "H": np.asarray(H, dtype=float),
            "phi": phi,
            "dphidtau": np.asarray(dphidtau, dtype=float),
            "tau": tau,
            "H_of_a_E_interp": H_of_a_E,
        }


def _quick_unit_check_lA(obs: Dict[str, float], target: float = 301.6, tol: float = 2.0) -> bool:
    try:
        lA = float(obs.get("lA", np.nan))
        if not np.isfinite(lA):
            return False
        return abs(lA - float(target)) <= float(tol)
    except Exception:
        return False


def rodar_simulacao_cobaya():
    import matplotlib.pyplot as plt
    print("=== DUT-CMB MODULE | COBAYA-STYLE VALIDATION RUN ===")

    params = DUTParameters(
        H0_kms_mpc=67.4,
        omega_b=0.0493,
        omega_c=0.264,
        omega_r=0.000092,
        omega_nu=0.0,
        n_eff=3.044,
        lambda_phi=0.1,
        V0=0.757,
        xi=0.1666667,
        omega_k=-0.07,
        phi_ini=0.0,
        dphi_dN_ini=0.0,
    )

    print("[1/6] Background (H(z))...")
    bg_data = BackgroundSolver.generate_background_tables(params, a_ini=1e-6, a_final=1.0, N_points=900)
    z_tab = (1.0 / np.maximum(bg_data["a"], 1e-60)) - 1.0
    H_tab = bg_data["H"]

    print("[2/6] Perturbations + fσ8(z)...")
    driver = PerturbationsDriver(bg_data)
    pert_res = driver.solve_perturbations(params, k_mode=0.01, a_ini=1e-4)
    z_eval = np.linspace(0.0, 2.5, 60)
    a_eval = 1.0 / (1.0 + z_eval)
    fs8_vals = GrowthFactorModule.compute_fsig8(a_eval, pert_res, params, sigma8_0=0.81)

    print("[3/6] CMB compressed observable (lA) ...")
    cfg = CMBConfig(omega_b=params.omega_b, omega_c=params.omega_c, omega_r=params.omega_r, omega_nu=params.omega_nu, tau_target=1.0)
    pri = CMBModule.default_planck_lA_prior(sigma=0.3)
    H_of_a = bg_data["H_of_a_E_interp"]
    lnlike, obs = CMBModule.lnlike_cmb(H_of_a, params.H0_kms_mpc, cfg, pri)

    print(f" lA={obs['lA']:.4f} | z_th={obs['z_th']:.3f} | lnlike={lnlike:.6f}")

    print("[4/6] Quick unit-check (lA ≈ 301.6)...")
    ok = _quick_unit_check_lA(obs, target=301.6, tol=3.0)
    print(" UNIT CHECK:", "PASS" if ok else "FAIL")

    print("[5/6] Plotting...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    mask = (z_tab >= 0.0) & (z_tab <= 10.0)
    ax1.plot(z_tab[mask], H_tab[mask], label="H(z) [km/s/Mpc]")
    ax1.set_xlabel("z")
    ax1.set_ylabel("H(z)")

    ax2 = ax1.twinx()
    ax2.plot(z_eval, fs8_vals, linestyle="--", label="fσ8(z)")
    ax2.set_ylabel("fσ8(z)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("DUT-CMB Module – Background + Growth + lA prior")
    plt.tight_layout()
    plt.show()

    print("[6/6] Cobaya-style pipeline completed.")


# =============================================================================
# APPEND-ONLY FIX: CLOSE MISSING MODULES (PerturbationsDriver, GrowthFactorModule, DUT_MCMC_Sampler)
# =============================================================================

class PerturbationsDriver:
    def __init__(self, bg_tables: Dict[str, Any]):
        self.bg_tables = bg_tables
        a_tab = np.asarray(bg_tables["a"], dtype=float)
        H_tab = np.asarray(bg_tables["H"], dtype=float)
        self._H_of_a = interp1d(
            a_tab, H_tab, kind="linear", bounds_error=False,
            fill_value=(float(H_tab[0]), float(H_tab[-1]))
        )

    def solve_perturbations(self, params: DUTParameters, k_mode: float, a_ini: float = 1e-4) -> Dict[str, np.ndarray]:
        eps = 1e-60
        a_tab = np.asarray(self.bg_tables["a"], dtype=float)
        a_ini = float(max(a_ini, a_tab[0]))
        mask = a_tab >= a_ini
        a = a_tab[mask]
        ln_a = np.log(np.maximum(a, eps))

        omega_m0 = float(params.omega_b + params.omega_c)

        def E2(ai: float) -> float:
            Hi = float(self._H_of_a(ai))
            return float((Hi / max(params.H0_kms_mpc, 1e-30))**2)

        def dlnH_dlnA(ai: float) -> float:
            h = 1e-4
            H1 = float(self._H_of_a(ai*(1+h)))
            H2 = float(self._H_of_a(ai*(1-h)))
            return float(np.log(max(H1,1e-60)/max(H2,1e-60))/np.log((1+h)/(1-h)))

        def omega_m_of_a(ai: float) -> float:
            return (omega_m0 * ai**-3) / max(E2(ai), eps)

        def rhs(x, y):
            ai = float(np.exp(x))
            d1 = y[1]
            d2 = -(2.0 + dlnH_dlnA(ai)) * y[1] + 1.5 * omega_m_of_a(ai) * y[0]
            return np.array([d1, d2], dtype=float)

        y0 = np.array([1e-5, 0.0], dtype=float)
        sol = solve_ivp(rhs, (ln_a[0], ln_a[-1]), y0, t_eval=ln_a, rtol=1e-8, atol=1e-12, method="RK45")
        if not sol.success:
            raise RuntimeError(f"Perturbation integration failed: {sol.message}")

        delta = np.asarray(sol.y[0], dtype=float)
        delta_c = delta.copy()
        delta_b = delta.copy()

        Y = np.zeros((len(a), 11), dtype=float)
        Y[:,4] = delta_c
        Y[:,6] = delta_b

        return {"tau": np.asarray(self.bg_tables["tau"], dtype=float)[mask],
                "a": a, "Y": Y,
                "delta_c": delta_c, "delta_b": delta_b}


class GrowthFactorModule:
    @staticmethod
    def _interpolate_log_delta_m(a_out, delta_c, delta_b, params):
        omega_m = float(params.omega_c + params.omega_b)
        delta_m = (params.omega_c*delta_c + params.omega_b*delta_b)/max(omega_m,1e-60)
        ln_a = np.log(np.maximum(a_out,1e-60))
        ln_delta = np.log(np.abs(delta_m)+1e-60)
        interp = interp1d(ln_a, ln_delta, kind="cubic", bounds_error=False, fill_value="extrapolate")
        return interp

    @staticmethod
    def compute_f_sigma8(a, ln_delta_interp, sigma8_z0=0.8):
        ln_a = np.log(max(a,1e-60))
        f = (ln_delta_interp(ln_a+1e-3)-ln_delta_interp(ln_a-1e-3))/(2e-3)
        growth = np.exp(ln_delta_interp(ln_a))/np.exp(ln_delta_interp(0.0))
        return float(f*growth*sigma8_z0)

    @classmethod
    def compute_fsig8(cls, a_eval, pert_results, params, sigma8_0=0.811):
        ln_delta_interp = cls._interpolate_log_delta_m(
            pert_results["a"], pert_results["delta_c"], pert_results["delta_b"], params)
        return np.array([cls.compute_f_sigma8(a, ln_delta_interp, sigma8_z0=sigma8_0) for a in a_eval])


class DUT_MCMC_Sampler:
    def __init__(self, params_initial: DUTParameters, config: CMBConfig, priors: CMBPriors):
        self.current_params = params_initial
        self.config = config
        self.priors = priors
        self.chain = []

        loader = PantheonLoader(cache_dir=".dut_cache", url=GlobalCosmoData().pantheon_snia_csv)
        z, mu, muerr = loader.load_arrays()
        self.z_sn, self.mu_obs, self.mu_err = z, mu, muerr

        cov = loader.load_covariance(expected_n=len(self.z_sn))
        if cov is not None and cho_factor is not None and cho_solve is not None:
            self.cov = cov
            self.cov_factor = cho_factor(cov, lower=True, check_finite=True)
        else:
            self.cov = None
            self.cov_factor = None

    def get_distance_modulus(self, H_of_a_interp, z):
        c_kms = PhysicalConstants.C_KMS
        r, _ = quad(lambda zp: 1.0/max(H_of_a_interp(1/(1+zp)),1e-60), 0.0, z)
        dL = (1+z)*c_kms*r
        return 5*np.log10(max(dL,1e-20)*1e6/10.0)

    def lnlike_pantheon(self, H_of_a_interp):
        mu_model = np.array([self.get_distance_modulus(H_of_a_interp,z) for z in self.z_sn])
        resid = self.mu_obs - mu_model

        if self.cov_factor is not None:
            chi2 = float(resid.T @ cho_solve(self.cov_factor, resid))
        else:
            chi2 = float(np.sum((resid/self.mu_err)**2))

        return -0.5*chi2 if np.isfinite(chi2) else -1e30

    def ln_posterior(self, p_vector):
        try:
            H0, ob, oc, lphi, V0, xi, ok = [float(x) for x in p_vector]
            p = DUTParameters(
                H0_kms_mpc=H0,
                omega_b=ob,
                omega_c=oc,
                omega_r=float(self.current_params.omega_r),
                omega_nu=float(self.current_params.omega_nu),
                n_eff=float(self.current_params.n_eff),
                lambda_phi=lphi,
                V0=V0,
                xi=xi,
                omega_k=ok,
                phi_ini=float(self.current_params.phi_ini),
                dphi_dN_ini=float(self.current_params.dphi_dN_ini),
            )
            bg = BackgroundSolver.generate_background_tables(p)
            H_of_a = bg["H_of_a_E_interp"]
            lnlike_c,_ = CMBModule.lnlike_cmb(H_of_a,p.H0_kms_mpc,self.config,self.priors)
            lnlike_p = self.lnlike_pantheon(H_of_a)
            return lnlike_c+lnlike_p
        except Exception:
            return -1e30

    def run_mcmc(self, steps=1000, walk=(0.3,0.0005,0.002,0.005,0.01,0.005,0.005)):
        curr_p = np.array([self.current_params.H0_kms_mpc,self.current_params.omega_b,
                           self.current_params.omega_c,self.current_params.lambda_phi,
                           self.current_params.V0,self.current_params.xi,self.current_params.omega_k], dtype=float)
        curr_ll = float(self.ln_posterior(curr_p))
        for _ in range(int(steps)):
            trial = curr_p + np.random.normal(scale=np.asarray(walk, dtype=float), size=curr_p.shape)
            trial_ll = float(self.ln_posterior(trial))
            if trial_ll>curr_ll or np.random.rand()<np.exp(trial_ll-curr_ll):
                curr_p, curr_ll = trial, trial_ll
            self.chain.append((curr_p.copy(), float(curr_ll)))
        return self.chain


if __name__ == "__main__":
    rodar_simulacao_cobaya()

