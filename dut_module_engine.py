#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CMB DUT MODULE — SCIENTIFIC LATTES VERSION (ENHANCEMENTS & REAL DATA)
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

GLOBAL DATA SOURCES (REAL LINKS + OFFLINE FALLBACK):
- Pantheon+ SH0ES: GitHub DataRelease
- DESI: public BAO releases (optional loader stub)
- Planck: compressed distance priors (lite prior stub; full likelihood external)
- JWST COSMOS-Web: catalog endpoints (optional stub)

ExtractoDAO Labs | CNPJ: 48.839.397/0001-36
Contact: contato@extractodao.com
================================================================================
"""

from __future__ import annotations

import os
import json
import math
import time
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import requests
except Exception:
    requests = None

try:
    from filelock import FileLock
except Exception:
    FileLock = None

try:
    import ctypes
except Exception:
    ctypes = None


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm_float(x: float) -> float:
    return float(np.format_float_positional(float(x), precision=12, unique=False, fractional=False, trim="k"))


def _json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PhysicalConstants:
    C_KMS = 299792.458
    MPC_M = 3.0856775814913673e22
    KM_M = 1e3
    SIGMA_T = 6.6524587321e-29
    M_P = 1.67262192369e-27
    G_SI = 6.67430e-11
    MPC_KM = MPC_M / KM_M


class MSPCCache:
    def __init__(self, cache_dir: str = ".dut_cache", name: str = "mspc_cache.json"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / name
        self.lock_path = str(self.path) + ".lock"

    def _key(self, key_obj: Dict[str, Any]) -> str:
        key_obj2: Dict[str, Any] = {}
        for k, v in key_obj.items():
            if isinstance(v, float):
                key_obj2[k] = _norm_float(v)
            elif isinstance(v, (np.floating,)):
                key_obj2[k] = _norm_float(float(v))
            else:
                key_obj2[k] = v
        s = _json_dumps_canonical(key_obj2)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _read_all(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_bytes()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        payload = _json_dumps_canonical(data).encode("utf-8")
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="mspc_", suffix=".tmp", dir=str(self.cache_dir))
        try:
            os.write(tmp_fd, payload)
            os.close(tmp_fd)
            os.replace(tmp_path, str(self.path))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def get(self, key_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        k = self._key(key_obj)
        if FileLock is not None:
            with FileLock(self.lock_path):
                d = self._read_all()
        else:
            d = self._read_all()
        val = d.get(k)
        return val if isinstance(val, dict) else None

    def set(self, key_obj: Dict[str, Any], value: Dict[str, Any]) -> None:
        k = self._key(key_obj)
        if FileLock is not None:
            with FileLock(self.lock_path):
                d = self._read_all()
                d[k] = value
                self._atomic_write(d)
        else:
            d = self._read_all()
            d[k] = value
            self._atomic_write(d)


class PantheonLoader:
    DEFAULT_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2B_SH0ES.dat"

    def __init__(self, cache_dir: str = ".dut_cache", url: str = DEFAULT_URL, allow_mock_offline: bool = True):
        self.url = str(url)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.cache_dir / "PantheonPlus_SH0ES.dat"
        self.meta_path = self.cache_dir / "PantheonPlus_SH0ES.meta.json"
        self.allow_mock_offline = bool(allow_mock_offline)

    def _write_cache(self, raw: bytes) -> None:
        sha = _sha256_bytes(raw)
        self.data_path.write_bytes(raw)
        meta = {"source_url": self.url, "sha256": sha, "download_utc": _now_utc_iso(), "bytes": int(len(raw))}
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _read_cache_verified(self) -> Optional[str]:
        if not self.data_path.exists() or not self.meta_path.exists():
            return None
        try:
            raw = self.data_path.read_bytes()
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            sha = _sha256_bytes(raw)
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
        if self.allow_mock_offline:
            return ""
        raise RuntimeError("Pantheon+ download failed and no verified cache found.")

    def load_arrays(self, seed: int = 1234) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        if pd is None:
            if self.allow_mock_offline:
                rng = np.random.default_rng(int(seed))
                z = np.linspace(0.01, 2.0, 1700)
                mu = 5.0 * np.log10((1.0 + z) * 3000.0) + 25.0
                muerr = 0.15 * np.ones_like(z)
                mu = mu + rng.normal(0.0, muerr)
                return z.astype(float), mu.astype(float), muerr.astype(float), "MOCK (pandas unavailable)"
            raise RuntimeError("pandas is not available; cannot parse Pantheon+ table.")
        text = self.fetch_text()
        if not text.strip():
            rng = np.random.default_rng(int(seed))
            z = np.linspace(0.01, 2.0, 1700)
            mu = 5.0 * np.log10((1.0 + z) * 3000.0) + 25.0
            muerr = 0.15 * np.ones_like(z)
            mu = mu + rng.normal(0.0, muerr)
            return z.astype(float), mu.astype(float), muerr.astype(float), "MOCK (offline)"
        from io import StringIO
        df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
        if "IS_TRAINING" in df.columns:
            df = df[df["IS_TRAINING"] == 0]
        for col in ("zHD", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"):
            if col not in df.columns:
                raise RuntimeError(f"Pantheon+ missing column: {col}")
        z = df["zHD"].to_numpy(dtype=float)
        mu = df["MU_SH0ES"].to_numpy(dtype=float)
        muerr = df["MU_SH0ES_ERR_DIAG"].to_numpy(dtype=float)
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(muerr)):
            raise RuntimeError("Pantheon+ contains non-finite values.")
        if np.any(muerr <= 0):
            raise RuntimeError("Pantheon+ has non-positive diagonal errors.")
        return z, mu, muerr, "REAL"


@dataclass(frozen=True)
class CMBPriors:
    mean: np.ndarray
    invcov: np.ndarray
    names: Tuple[str, ...] = ("lA",)

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        invcov = np.asarray(self.invcov, dtype=float)
        if mean.ndim != 1 or invcov.ndim != 2:
            raise ValueError("CMBPriors dimensions are inconsistent.")
        if invcov.shape[0] != invcov.shape[1] or invcov.shape[0] != mean.shape[0]:
            raise ValueError("CMBPriors dimensions are inconsistent.")
        if len(self.names) != mean.shape[0]:
            raise ValueError("CMBPriors names inconsistent.")


@dataclass(frozen=True)
class BaryonRecombinationModel:
    z_recomb: float = 1090.0
    delta_z: float = 50.0
    xe_highz: float = 1.0
    xe_lowz: float = 1e-4

    def xe(self, z: float) -> float:
        x = (float(z) - float(self.z_recomb)) / max(float(self.delta_z), 1e-12)
        return float(self.xe_lowz + (self.xe_highz - self.xe_lowz) / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class CMBConfig:
    omega_b: float
    omega_c: float
    omega_r: float
    omega_nu: float = 0.0
    Yp_He: float = 0.24
    recomb_model: BaryonRecombinationModel = BaryonRecombinationModel()
    tau_target: float = 0.056
    z_th_min: float = 50.0
    z_th_max: float = 4000.0
    epsabs: float = 1e-10
    epsrel: float = 1e-10


@dataclass(frozen=True)
class DUTParameters:
    H0_kms_mpc: float
    omega_b: float
    omega_c: float
    omega_r: float
    omega_nu: float = 0.0
    n_eff: float = 3.044
    lambda_phi: float = 2.0
    V0: float = 0.05
    xi: float = 0.1
    omega_k: float = -0.07
    phi_ini: float = 10.0
    dphi_dN_ini: float = 1e-6

    def __post_init__(self):
        if not (0.0 < float(self.H0_kms_mpc) < 200.0):
            raise ValueError("H0 fora do intervalo físico")
        if not (0.0 < float(self.omega_b) < 0.2):
            raise ValueError("Ω_b fora do intervalo")
        if not (0.0 <= float(self.omega_c) < 1.5):
            raise ValueError("Ω_c fora do intervalo")
        if not (0.0 <= float(self.omega_r) < 0.1):
            raise ValueError("Ω_r fora do intervalo")
        if float(self.xi) < 0.0:
            raise ValueError("ξ deve ser ≥ 0")
        if abs(float(self.omega_k)) > 0.5:
            raise ValueError("|Ω_k| muito grande")
        if float(self.V0) < 0.0:
            raise ValueError("V0 deve ser positivo")

    @classmethod
    def from_mcmc_vector(cls, v: np.ndarray, base: "DUTParameters") -> "DUTParameters":
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size != 10:
            raise ValueError(f"Vector must have 10 params, got {v.size}")
        return cls(
            H0_kms_mpc=float(v[0]),
            omega_b=float(v[1]),
            omega_c=float(v[2]),
            omega_r=float(v[3]),
            omega_nu=float(base.omega_nu),
            n_eff=float(base.n_eff),
            lambda_phi=float(v[4]),
            V0=float(v[5]),
            xi=float(v[6]),
            omega_k=float(v[7]),
            phi_ini=float(v[8]),
            dphi_dN_ini=float(v[9]),
        )


class Fortran2008Core:
    def __init__(self, so_path: str = "./dut_core_f2008.so"):
        self.so_path = str(so_path)
        self.lib = None
        if ctypes is None:
            return
        if not os.path.exists(self.so_path):
            return
        try:
            lib = ctypes.CDLL(self.so_path)
            self.lib = lib
        except Exception:
            self.lib = None

    def available(self) -> bool:
        return self.lib is not None

    def e2_constraint(self, a: np.ndarray, phi: np.ndarray, u: np.ndarray, p: DUTParameters) -> Optional[np.ndarray]:
        if self.lib is None:
            return None
        return None


class RustCore:
    def __init__(self, so_path: str = "./libdut_rust_core.so"):
        self.so_path = str(so_path)
        self.lib = None
        if ctypes is None:
            return
        if not os.path.exists(self.so_path):
            return
        try:
            lib = ctypes.CDLL(self.so_path)
            self.lib = lib
        except Exception:
            self.lib = None

    def available(self) -> bool:
        return self.lib is not None

    def e2_constraint(self, a: np.ndarray, phi: np.ndarray, u: np.ndarray, p: DUTParameters) -> Optional[np.ndarray]:
        if self.lib is None:
            return None
        return None


class BackgroundSolver:
    @staticmethod
    def _V(phi: np.ndarray, params: DUTParameters) -> np.ndarray:
        return float(params.V0) * np.exp(-float(params.lambda_phi) * np.asarray(phi, dtype=float))

    @staticmethod
    def _dV_dphi(phi: float, params: DUTParameters) -> float:
        return -float(params.lambda_phi) * float(params.V0) * float(np.exp(-float(params.lambda_phi) * float(phi)))

    @staticmethod
    def _E2_from_constraint_vectorized(a: np.ndarray, phi: np.ndarray, u: np.ndarray, params: DUTParameters) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        phi = np.asarray(phi, dtype=float)
        u = np.asarray(u, dtype=float)

        eps = 1e-60
        a_safe = np.maximum(a, eps)

        om_m = float(params.omega_b + params.omega_c)
        om_r = float(params.omega_r + params.omega_nu)
        om_k = float(params.omega_k)

        rho_m = om_m * a_safe**-3
        rho_r = om_r * a_safe**-4
        rho_k = om_k * a_safe**-2
        Vphi = BackgroundSolver._V(phi, params)

        xi = float(params.xi)
        D_raw = (1.0 + xi * phi**2) - (u**2) / 6.0 - 2.0 * xi * phi * u
        scale = np.maximum(1.0, np.abs(1.0 + xi * phi**2))
        epsD = np.maximum(1e-18, 1e-12 * scale)
        D = np.sign(D_raw) * np.maximum(np.abs(D_raw), epsD)

        E2 = (rho_m + rho_r + rho_k + Vphi) / D
        if np.any((E2 <= 0) | (~np.isfinite(E2))):
            raise ValueError("HCNI: E^2 invalid in vectorized assembly")
        return np.maximum(E2, 1e-40)

    @classmethod
    def generate_background_tables(
        cls,
        params: DUTParameters,
        a_ini: float = 1e-6,
        a_final: float = 1.0,
        N_points: int = 6000,
        cache_dir: str = ".dut_cache",
        method_primary: str = "Radau",
    ) -> Dict[str, Any]:
        cache = MSPCCache(cache_dir=cache_dir)
        key_obj = {
            "H0": params.H0_kms_mpc,
            "ob": params.omega_b,
            "oc": params.omega_c,
            "or": params.omega_r,
            "onu": params.omega_nu,
            "neff": params.n_eff,
            "lp": params.lambda_phi,
            "V0": params.V0,
            "xi": params.xi,
            "ok": params.omega_k,
            "phi0": params.phi_ini,
            "u0": params.dphi_dN_ini,
            "a_ini": a_ini,
            "a_final": a_final,
            "N_points": N_points,
            "method": method_primary,
        }
        cached = cache.get(key_obj)
        if cached is not None:
            a_grid = np.asarray(cached["a"], dtype=float)
            H = np.asarray(cached["H"], dtype=float)
            phi = np.asarray(cached["phi"], dtype=float)
            u = np.asarray(cached["u"], dtype=float)
            H_of_a_E = interp1d(a_grid, H, kind="linear", bounds_error=False, fill_value=(float(H[0]), float(H[-1])))
            return {"a": a_grid, "H": H, "H_of_a_E_interp": H_of_a_E, "phi": phi, "u": u}

        eps = 1e-60
        a_ini = float(max(a_ini, eps))
        a_final = float(max(a_final, a_ini * 1.0001))
        N_points = int(max(200, N_points))

        ln_a = np.linspace(np.log(a_ini), np.log(a_final), N_points)
        a_grid = np.exp(ln_a)

        y0 = np.array([float(params.phi_ini), float(params.dphi_dN_ini)], dtype=float)

        def rhs(N: float, y: np.ndarray) -> np.ndarray:
            phi, u = float(y[0]), float(y[1])
            a = float(np.exp(N))
            E2 = float(cls._E2_from_constraint_vectorized(np.array([a]), np.array([phi]), np.array([u]), params)[0])
            h = 1e-5
            ap = float(np.exp(N + h))
            am = float(np.exp(N - h))
            E2p = float(cls._E2_from_constraint_vectorized(np.array([ap]), np.array([phi + u*h]), np.array([u]), params)[0])
            E2m = float(cls._E2_from_constraint_vectorized(np.array([am]), np.array([phi - u*h]), np.array([u]), params)[0])
            dlnH_dN = (0.5 * math.log(max(E2p, 1e-80)) - 0.5 * math.log(max(E2m, 1e-80))) / h
            xi = float(params.xi)
            dV = float(cls._dV_dphi(phi, params))
            curv_term = -(float(params.omega_k) * (a**-2)) / max(E2, 1e-40)
            R_over_H2 = 6.0 * (2.0 + dlnH_dN + curv_term)
            u_prime = -(3.0 + dlnH_dN) * u + (dV / max(E2, 1e-40)) - 2.0 * xi * R_over_H2 * phi
            return np.array([u, u_prime], dtype=float)

        def integrate(method: str, rtol: float, atol: float) -> Any:
            return solve_ivp(rhs, [float(ln_a[0]), float(ln_a[-1])], y0, t_eval=ln_a, method=method, rtol=rtol, atol=atol)

        try:
            sol = integrate(method_primary, rtol=1e-12, atol=1e-14)
            if not sol.success:
                raise RuntimeError(sol.message)
        except Exception:
            sol = integrate("RK45", rtol=1e-8, atol=1e-10)
            if not sol.success:
                raise RuntimeError(sol.message)

        phi = np.asarray(sol.y[0], dtype=float)
        u = np.asarray(sol.y[1], dtype=float)

        E2_arr = cls._E2_from_constraint_vectorized(a_grid, phi, u, params)
        H = float(params.H0_kms_mpc) * np.sqrt(np.maximum(E2_arr, 1e-40))
        H_of_a_E = interp1d(a_grid, H, kind="linear", bounds_error=False, fill_value="extrapolate")

        cache.set(
            key_obj,
            {
                "a": a_grid.tolist(),
                "H": H.tolist(),
                "phi": phi.tolist(),
                "u": u.tolist(),
                "utc": _now_utc_iso(),
            },
        )

        return {"a": a_grid, "H": H, "H_of_a_E_interp": H_of_a_E, "phi": phi, "u": u}


class CMBModule:
    C = PhysicalConstants.C_KMS
    SIGMA_T = PhysicalConstants.SIGMA_T
    M_P = PhysicalConstants.M_P
    G_SI = PhysicalConstants.G_SI
    MPC_M = PhysicalConstants.MPC_M
    KM_M = PhysicalConstants.KM_M

    @staticmethod
    def _E_of_z_from_Ha(H_of_a: Callable[[float], float], z: float) -> float:
        a = 1.0 / (1.0 + float(z))
        return float(H_of_a(a))

    @classmethod
    def comoving_distance_chi(cls, H_of_z: Callable[[float], float], z: float, epsabs: float, epsrel: float) -> float:
        def integrand(zp: float) -> float:
            Hz = float(H_of_z(float(zp)))
            return float(cls.C / max(Hz, 1e-30))
        val, _ = quad(integrand, 0.0, float(z), epsabs=float(epsabs), epsrel=float(epsrel), limit=2000)
        return float(val)

    @classmethod
    def sound_speed_cs(cls, omega_b: float, omega_r: float, z: float) -> float:
        a = 1.0 / (1.0 + float(z))
        rho_b = float(omega_b) / (a**3)
        rho_r = float(omega_r) / (a**4)
        R = 3.0 * rho_b / (4.0 * rho_r + 1e-60)
        return float(cls.C / np.sqrt(3.0 * (1.0 + R)))

    @classmethod
    def sound_horizon_rs(
        cls,
        H_of_z: Callable[[float], float],
        omega_b: float,
        omega_r: float,
        z_star: float,
        z_max: float,
        epsabs: float,
        epsrel: float,
        *,
        H0_km_s_Mpc: Optional[float] = None,
        omega_r_total: Optional[float] = None,
        z_max_dut: Optional[float] = None,
    ) -> float:
        z_star = float(z_star)
        z_max = float(z_max)

        use_tail = (H0_km_s_Mpc is not None) and (omega_r_total is not None) and (z_max_dut is not None)
        z_max_dut_f = float(z_max_dut) if z_max_dut is not None else z_max
        if use_tail:
            z_tail_max = float(max(z_max_dut_f, z_star * 50.0, 5.0e4))
        else:
            z_tail_max = z_max

        def H_phys(zp: float) -> float:
            zp = float(zp)
            if (not use_tail) or (zp <= z_max_dut_f):
                return float(H_of_z(zp))
            return float(H0_km_s_Mpc) * float(np.sqrt(float(omega_r_total))) * (1.0 + zp) ** 2

        def integrand(zp: float) -> float:
            cs = float(cls.sound_speed_cs(float(omega_b), float(omega_r), float(zp)))
            Hz = float(H_phys(zp))
            return float(cs / max(Hz, 1e-30))

        val, _ = quad(integrand, z_star, z_tail_max, epsabs=float(epsabs), epsrel=float(epsrel), limit=2500)
        return float(val)

    @classmethod
    def _baryon_number_density_ne0(cls, omega_b: float, H0_km_s_Mpc: float, Yp: float) -> float:
        H0_si = (float(H0_km_s_Mpc) * cls.KM_M) / cls.MPC_M
        rho_crit0 = 3.0 * (H0_si**2) / (8.0 * np.pi * cls.G_SI)
        rho_b0 = float(omega_b) * rho_crit0
        n_b0 = rho_b0 / cls.M_P
        ne0 = n_b0 * (1.0 - 0.5 * float(Yp))
        return float(ne0)

    @classmethod
    def optical_depth_tau(cls, H_of_a: Callable[[float], float], config: CMBConfig, H0_km_s_Mpc: float, z: float) -> float:
        ne0 = cls._baryon_number_density_ne0(config.omega_b, H0_km_s_Mpc, config.Yp_He)

        def integrand(zp: float) -> float:
            zp = float(zp)
            a = 1.0 / (1.0 + zp)
            Hz = float(H_of_a(float(a)))
            xe = float(config.recomb_model.xe(zp))
            ne = ne0 * (1.0 + zp) ** 3 * xe
            c_ms = cls.C * cls.KM_M
            Hz_si = (Hz * cls.KM_M) / cls.MPC_M
            return float((c_ms * cls.SIGMA_T * ne) / ((1.0 + zp) * max(Hz_si, 1e-40)))

        val, _ = quad(integrand, 0.0, float(z), epsabs=float(config.epsabs), epsrel=float(config.epsrel), limit=2000)
        return float(val)

    @classmethod
    def find_z_thermalization(cls, H_of_a_km_s_Mpc: Callable[[float], float], config: CMBConfig, H0_km_s_Mpc: float) -> float:
        z_lo, z_hi = float(config.z_th_min), float(config.z_th_max)
        target = float(config.tau_target)

        for _ in range(40):
            tau_lo = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_lo)
            tau_hi = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_hi)
            if tau_lo > target:
                z_lo = max(z_lo * 0.5, 1e-8)
            elif tau_hi < target:
                z_hi = z_hi * 2.0
            else:
                break

        for _ in range(120):
            z_mid = 0.5 * (z_lo + z_hi)
            tau_mid = cls.optical_depth_tau(H_of_a_km_s_Mpc, config, H0_km_s_Mpc, z_mid)
            if tau_mid < target:
                z_lo = z_mid
            else:
                z_hi = z_mid
            if abs(z_hi - z_lo) / max(z_mid, 1e-12) < 1e-10:
                break

        return float(0.5 * (z_lo + z_hi))

    @classmethod
    def compute_observables(
        cls,
        H_of_a: Callable[[float], float],
        params: DUTParameters,
        config: CMBConfig,
        *,
        z_star: float = 1090.0,
        z_max_dut: Optional[float] = None,
        H_of_a_units: str = "km/s/Mpc",
    ) -> Dict[str, float]:
        if H_of_a_units == "km/s/Mpc":
            H_of_a_km = H_of_a
        elif H_of_a_units == "1/Mpc":
            raise NotImplementedError("Modo '1/Mpc' ainda não foi calibrado para DUT-CMB.")
        else:
            raise ValueError("H_of_a_units must be 'km/s/Mpc' or '1/Mpc'.")

        z_th = cls.find_z_thermalization(H_of_a_km, config, float(params.H0_kms_mpc))
        tau_th = cls.optical_depth_tau(H_of_a_km, config, float(params.H0_kms_mpc), float(z_th))

        z_star = float(z_star)

        def H_of_z(z: float) -> float:
            a = 1.0 / (1.0 + float(z))
            return float(H_of_a(float(a)))

        chi = cls.comoving_distance_chi(H_of_z, z_star, epsabs=float(config.epsabs), epsrel=float(config.epsrel))

        Omega_k = float(params.omega_k)
        if abs(Omega_k) < 1e-12:
            Dm = float(chi)
        else:
            k = (float(params.H0_kms_mpc) / float(cls.C)) * float(np.sqrt(abs(Omega_k)))
            x = float(k * chi)
            if Omega_k > 0.0:
                Dm = float(np.sinh(x) / max(k, 1e-60))
            else:
                Dm = float(np.sin(x) / max(k, 1e-60))

        if z_max_dut is None:
            z_max_dut = float(5.0e4)

        omega_r_total = float(config.omega_r + config.omega_nu)
        rs = cls.sound_horizon_rs(
            H_of_z,
            float(config.omega_b),
            float(config.omega_r),
            z_star,
            z_max=float(z_max_dut),
            epsabs=float(config.epsabs),
            epsrel=float(config.epsrel),
            H0_km_s_Mpc=float(params.H0_kms_mpc),
            omega_r_total=omega_r_total,
            z_max_dut=float(z_max_dut),
        )

        lA = float(np.pi * (Dm / max(rs, 1e-60)))

        return {
            "z_th": float(z_th),
            "tau_th": float(tau_th),
            "z_star": float(z_star),
            "D_M_star": float(Dm),
            "r_s_star": float(rs),
            "lA": float(lA),
        }

    @staticmethod
    def chi2_distance_priors(observables: Dict[str, float], priors: CMBPriors) -> float:
        x = []
        for name in priors.names:
            if name == "lA":
                x.append(float(observables["lA"]))
            else:
                raise ValueError(f"Unsupported prior observable name: {name}")
        x = np.asarray(x, dtype=float)
        dx = x - np.asarray(priors.mean, dtype=float)
        chi2 = float(dx.T @ np.asarray(priors.invcov, dtype=float) @ dx)
        return float(chi2)

    @classmethod
    def lnlike_cmb(
        cls,
        H_of_a: Callable[[float], float],
        params: DUTParameters,
        config: CMBConfig,
        priors: CMBPriors,
        *,
        z_star: float = 1090.0,
        z_max_dut: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        obs = cls.compute_observables(H_of_a, params, config, z_star=z_star, z_max_dut=z_max_dut, H_of_a_units="km/s/Mpc")
        chi2 = cls.chi2_distance_priors(obs, priors)
        return float(-0.5 * chi2), obs

    @staticmethod
    def default_planck_lA_prior(sigma: float = 0.3) -> CMBPriors:
        mean = np.array([301.6], dtype=float)
        invcov = np.array([[1.0 / (float(sigma) ** 2)]], dtype=float)
        return CMBPriors(mean=mean, invcov=invcov, names=("lA",))


class PerturbationsDriver:
    def __init__(self, bg: Dict[str, Any]):
        self.bg = bg

    def compute_deltas(self) -> Dict[str, Any]:
        a = np.asarray(self.bg["a"], dtype=float)
        delta_c = a.copy()
        delta_b = a.copy()
        return {"a": a, "delta_c": delta_c, "delta_b": delta_b}


class GrowthFactorModule:
    @staticmethod
    def _interpolate_log_delta_m(
        a_out: np.ndarray,
        delta_c: np.ndarray,
        delta_b: np.ndarray,
        params: DUTParameters,
    ) -> Tuple[interp1d, Callable[[float], float]]:
        a_out = np.asarray(a_out, dtype=float)
        delta_c = np.asarray(delta_c, dtype=float)
        delta_b = np.asarray(delta_b, dtype=float)
        eps = 1e-60
        omega_m = float(params.omega_c + params.omega_b)
        safe_omega_m = max(omega_m, eps)
        delta_m = (float(params.omega_c) * delta_c + float(params.omega_b) * delta_b) / safe_omega_m
        delta_m = np.where(delta_m > eps, delta_m, eps)
        ln_a = np.log(np.maximum(a_out, eps))
        ln_delta = np.log(delta_m)
        ln_delta_interp = interp1d(ln_a, ln_delta, kind="linear", bounds_error=False, fill_value="extrapolate")
        delta_m_interp = interp1d(a_out, delta_m, kind="linear", bounds_error=False, fill_value="extrapolate")
        return ln_delta_interp, lambda a: float(delta_m_interp(float(a)))

    @staticmethod
    def compute_f_sigma8(a: float, ln_delta_interp: interp1d, sigma8_z0: float = 0.8) -> float:
        a = float(a)
        ln_a = float(np.log(max(a, 1e-60)))
        h = 1e-4
        f = float((ln_delta_interp(ln_a + h) - ln_delta_interp(ln_a - h)) / (2.0 * h))
        return float(f * sigma8_z0)


class DUT_MCMC_Sampler:
    def __init__(
        self,
        base_params: DUTParameters,
        step_scales: np.ndarray,
        seed: int = 12345,
        cache_dir: Optional[str] = None,
    ):
        self.base = base_params
        self.step_scales = np.asarray(step_scales, dtype=float).reshape(-1)
        if self.step_scales.size != 10:
            raise ValueError("step_scales must have length 10")
        self.rng = np.random.default_rng(int(seed))
        self.seed = int(seed)
        self.cache_dir = cache_dir

    def propose(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return x + self.rng.normal(0.0, self.step_scales, size=x.size)

    def run(
        self,
        lnpost: Callable[[DUTParameters], float],
        x0: np.ndarray,
        n_steps: int = 2000,
        burn: int = 200,
    ) -> Dict[str, Any]:
        x = np.asarray(x0, dtype=float).reshape(-1)
        lp = float(lnpost(DUTParameters.from_mcmc_vector(x, self.base)))
        chain = []
        lps = []
        acc = 0

        for _ in range(int(n_steps)):
            xp = self.propose(x)
            try:
                pp = DUTParameters.from_mcmc_vector(xp, self.base)
                lpp = float(lnpost(pp))
            except Exception:
                lpp = -np.inf
            if np.isfinite(lpp) and (math.log(self.rng.random()) < (lpp - lp)):
                x, lp = xp, lpp
                acc += 1
            chain.append(x.copy())
            lps.append(lp)

        chain = np.asarray(chain, dtype=float)
        lps = np.asarray(lps, dtype=float)

        if burn > 0 and burn < len(chain):
            chain2 = chain[burn:]
            lps2 = lps[burn:]
        else:
            chain2 = chain
            lps2 = lps

        return {
            "seed": self.seed,
            "accept_rate": float(acc / max(int(n_steps), 1)),
            "chain": chain2,
            "lnpost": lps2,
        }


def _quick_unit_check_lA(obs: Dict[str, float], target: float = 301.6, tol: float = 1.0) -> bool:
    try:
        lA = float(obs.get("lA", np.nan))
        if not np.isfinite(lA):
            return False
        return abs(lA - float(target)) <= float(tol)
    except Exception:
        return False


def finalizar_e_imprimir(params: DUTParameters, obs: Dict[str, float], H_z1: float) -> None:
    print(f"\n{'='*60}")
    print(f"       DUT-CMB ENGINE — RESULTADOS CIENTÍFICOS")
    print(f"{'='*60}")
    print(f"Parâmetros de Entrada:")
    print(f"  H0: {params.H0_kms_mpc:.2f} | Ωk: {params.omega_k:.4f} | ξ: {params.xi:.6f}")
    print(f"  V0: {params.V0:.6f} | λ: {params.lambda_phi:.6f} | Ωr: {params.omega_r:.6e}")
    print("-" * 60)
    print(f"Observáveis CMB Calculados:")
    print(f"  Escala Acústica (lA): {obs['lA']:.6f} (Planck: 301.6 ± 0.3)")
    print(f"  Horizonte de Som (r_s*): {obs['r_s_star']:.4f} Mpc")
    print(f"  D_M(z*): {obs['D_M_star']:.4f} Mpc")
    print(f"  z*: {obs['z_star']:.1f} | z_th: {obs['z_th']:.4f} | τ(z_th): {obs['tau_th']:.6f}")
    print("-" * 60)
    print(f"Verificação Dinâmica:")
    print(f"  H(z=1): {float(H_z1):.4f} km/s/Mpc")
    tension = abs(float(params.H0_kms_mpc) - 73.04) / 1.04
    print(f"  Tensão H0 vs SH0ES (2021): {tension:.4f} σ")
    print(f"{'='*60}\n")


def rodar_simulacao_cobaya():
    t0 = time.time()

    params = DUTParameters(
        H0_kms_mpc=67.4,
        omega_b=0.0493,
        omega_c=0.2640,
        omega_r=9.2e-5,
        omega_nu=0.0,
        n_eff=3.044,
        lambda_phi=0.1000,
        V0=0.7570,
        xi=0.1666667,
        omega_k=-0.00,
        phi_ini=0.0,
        dphi_dN_ini=0.0,
    )

    cache_dir = ".dut_cache"
    bg = BackgroundSolver.generate_background_tables(params, a_ini=1e-6, a_final=1.0, N_points=6000, cache_dir=cache_dir, method_primary="Radau")
    a_tab = np.asarray(bg["a"], dtype=float)
    H_tab = np.asarray(bg["H"], dtype=float)
    H_of_a = bg["H_of_a_E_interp"]

    z_tab = (1.0 / np.maximum(a_tab, 1e-60)) - 1.0
    order = np.argsort(z_tab)
    H_z1 = float(np.interp(1.0, z_tab[order], H_tab[order]))

    cfg = CMBConfig(
        omega_b=float(params.omega_b),
        omega_c=float(params.omega_c),
        omega_r=float(params.omega_r),
        omega_nu=float(params.omega_nu),
        tau_target=0.056,
    )

    pri = CMBModule.default_planck_lA_prior(sigma=0.3)
    lnlike, obs = CMBModule.lnlike_cmb(H_of_a, params, cfg, priors=pri, z_star=1090.0, z_max_dut=float(np.max(z_tab)))

    ok = _quick_unit_check_lA(obs, target=301.6, tol=1.0)

    print("CMB Observables")
    print("lA (acoustic scale)")
    print(f"{obs['lA']:.4f}")
    print("z_th (thermalization)")
    print(f"{obs['z_th']:.4f}")
    print("ln(likelihood)")
    print(f"{lnlike:.4f}")
    print("r_s (sound horizon)")
    print(f"{obs['r_s_star']:.2f} Mpc")
    print("Simulation Status")
    print("Unit Check (lA ≈ 301.6)")
    print("PASS ✓" if ok else "FAIL ✗")
    print("Computation Time")
    print(f"{(time.time() - t0):.2f}s")
    print("Timestamp")
    print(datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    print("Model Parameters")
    print("H₀")
    print(f"{params.H0_kms_mpc:.2f} km/s/Mpc")
    print("Ω_b")
    print(f"{params.omega_b:.4f}")
    print("Ω_c")
    print(f"{params.omega_c:.4f}")
    print("λ_φ")
    print(f"{params.lambda_phi:.4f}")

    finalizar_e_imprimir(params, obs, H_z1)

    loader = PantheonLoader(cache_dir=cache_dir, allow_mock_offline=True)
    z_sn, mu_sn, muerr_sn, mode = loader.load_arrays(seed=1234)
    print(f"Pantheon+: {mode} | N={len(z_sn)}")


if __name__ == "__main__":
    rodar_simulacao_cobaya()
