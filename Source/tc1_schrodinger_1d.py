#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TC1 – Schrödinger 1D (Diferencias finitas, Python)
# Autor: Axel (UNED – FC1)
# Requisitos: numpy, scipy, matplotlib

from dataclasses import dataclass
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import csv, os

# ==============================
# Configuración de directorios
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORIA_DIR = os.path.join(BASE_DIR, "../memoria/figuras")
EXTRAS_DIR  = os.path.join(BASE_DIR, "../extras")
os.makedirs(MEMORIA_DIR, exist_ok=True)
os.makedirs(EXTRAS_DIR, exist_ok=True)

# ==============================
# Utilidades numéricas
# ==============================

def make_grid(xmin: float, xmax: float, N: int):
    """Crea malla 1D uniforme (N puntos internos) con Dirichlet en extremos.
    Retorna x (N,), dx (float)."""
    x = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    return x, dx

def d2_operator_dirichlet(N: int, dx: float) -> sp.csr_matrix:
    """Segunda derivada centrada (Dirichlet implícitas), tridiagonal (CSR)."""
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N - 1)
    D2 = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N), format="csr")
    return D2 / (dx**2)

def d1_operator_central(N: int, dx: float) -> sp.csr_matrix:
    """Primera derivada centrada (útil para <p>), con ceros en extremos."""
    off = 0.5 / dx
    D1 = sp.diags([-off*np.ones(N-1), off*np.ones(N-1)], [-1, 1], shape=(N, N), format="csr")
    return D1

# ==============================
# Potenciales típicos
# ==============================

def V_infinite_well(x: np.ndarray, L: float, xmin: float) -> np.ndarray:
    """Pozo infinito numericamente: V=0 en [xmin, xmin+L], 
    las condiciones de contorno Dirichlet hacen el resto."""
    # Suponemos dominio elegido como [xmin, xmin+L] para claridad
    return np.zeros_like(x)

def V_finite_well(x: np.ndarray, V0: float, a: float, x0: float) -> np.ndarray:
    """Pozo finito rectangular de profundidad V0 (>0) y anchura a, centrado en x0."""
    V = np.zeros_like(x)
    mask = np.abs(x - x0) <= a/2.0
    V[mask] = -abs(V0)
    return V

def V_harmonic(x: np.ndarray, m: float, omega: float) -> np.ndarray:
    """Oscilador armónico 1D: (1/2) m ω^2 x^2."""
    return 0.5 * m * (omega**2) * (x**2)

# ==============================
# Hamiltoniano y solución
# ==============================

@dataclass
class Units:
    hbar: float = 1.0
    m: float    = 1.0

def hamiltonian_1d(x: np.ndarray, Vx: np.ndarray, units: Units, D2: sp.csr_matrix) -> sp.csr_matrix:
    """H = -(hbar^2 / 2m) D2 + diag(V)."""
    kin = -(units.hbar**2) / (2.0 * units.m) * D2
    Vop = sp.diags(Vx, 0, format="csr")
    return kin + Vop

def solve_eigen(H: sp.csr_matrix, k: int = 6):
    """Resuelve los k autovalores más bajos (simétrico) con eigsh."""
    # which='SA' (smallest algebraic) para matrices reales simétricas
    vals, vecs = spla.eigsh(H, k=k, which='SA')
    # Orden ascendente por si acaso
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

# ==============================
# Normalización y chequeos
# ==============================

def normalize_modes(psi: np.ndarray, dx: float) -> np.ndarray:
    """Normaliza columnas de psi tal que sum |psi|^2 dx = 1."""
    # psi: (N, k)
    norms = np.sqrt(np.sum(np.abs(psi)**2, axis=0) * dx)
    return psi / norms

def check_orthonormality(psi: np.ndarray, dx: float) -> np.ndarray:
    """Devuelve matriz de solapamiento S_ij = ∑ psi_i * psi_j dx (ideal ~ I)."""
    return (psi.T @ (psi * dx))

# ==============================
# Valores esperados (opcionales)
# ==============================

def expectation_x(x: np.ndarray, psi: np.ndarray, dx: float) -> float:
    return np.sum((x * (np.abs(psi)**2)) * dx)

def expectation_x2(x: np.ndarray, psi: np.ndarray, dx: float) -> float:
    return np.sum(((x**2) * (np.abs(psi)**2)) * dx)

def expectation_p2(psi: np.ndarray, D2: sp.csr_matrix, units: Units, dx: float) -> float:
    """<p^2> usando -ħ^2 d2/dx2, forma positiva / integración por partes."""
    # <p^2> = ⟨ψ | (-ħ^2 d2/dx2) | ψ⟩
    vec = D2 @ psi
    return - (units.hbar**2) * np.sum(np.conj(psi) * vec) * dx

def energies_from_expectations(psi: np.ndarray, H: sp.csr_matrix, dx: float) -> float:
    """Comprueba E = <ψ|H|ψ>."""
    Hpsi = H @ psi
    return np.real(np.sum(np.conj(psi) * Hpsi) * dx)

# ==============================
# Gráficas
# ==============================

def plot_potential_and_modes(x, Vx, vals, vecs, n_show=4, scale='auto', title=''):
    """Dibuja V(x) y las primeras autofunciones (re-escaladas y desplazadas por E_n)."""
    plt.figure(figsize=(8, 5))
    plt.plot(x, Vx, label='V(x)')
    # Escalado vertical de psi para que se vean
    if scale == 'auto':
        span = np.max(Vx) - np.min(Vx)
        scale = 0.1 * span if span > 0 else 1.0
    for n in range(min(n_show, vecs.shape[1])):
        psi = vecs[:, n]
        plt.plot(x, vals[n] + scale * psi / np.max(np.abs(psi)), label=f'n={n}')
    plt.xlabel('x')
    plt.ylabel('E, V')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# ==============================
# Analíticos para validar
# ==============================

def inf_well_energies_analytic(n, L, units: Units):
    """E_n pozo infinito [0,L]: (n^2 pi^2 ħ^2)/(2m L^2). n=1,2,..."""
    return (n**2) * (np.pi**2) * (units.hbar**2) / (2.0 * units.m * (L**2))

def ho_energies_analytic(n, omega, units: Units):
    """E_n oscilador: ħω (n + 1/2). n=0,1,2,..."""
    return units.hbar * omega * (n + 0.5)

# ==============================
# Demo / Casos de estudio
# ==============================

def demo_infinite_well():
    print("Caso 1: Pozo infinito (validación)")
    units = Units(hbar=1.0, m=1.0)
    L = 1.0
    N = 2000
    xmin, xmax = 0.0, L
    x, dx = make_grid(xmin, xmax, N)
    D2 = d2_operator_dirichlet(N, dx)
    Vx = V_infinite_well(x, L, xmin)
    H = hamiltonian_1d(x, Vx, units, D2)
    k = 6
    vals, vecs = solve_eigen(H, k=k)
    vecs = normalize_modes(vecs, dx)
    # Analítico: n = 1..k
    n = np.arange(1, k+1)
    E_anal = inf_well_energies_analytic(n, L, units)
    rel_err = (vals - E_anal)/E_anal
    print("E (num)  vs  E (anal)  y error relativo:")
    for i in range(k):
        print(f"n={n[i]:d}: {vals[i]:.6f}  |  {E_anal[i]:.6f}  |  {rel_err[i]:+.2e}")
    plot_potential_and_modes(x, Vx, vals, vecs, n_show=4, title='Pozo infinito')

    plt.savefig(os.path.join(MEMORIA_DIR, "fig_pozo_infinito.pdf"), bbox_inches="tight")

    with open(os.path.join(EXTRAS_DIR, "pozo_infinito_energies.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n", "E_num", "E_anal", "rel_error"])
        for i, (En, Ea) in enumerate(zip(vals, E_anal), start=1):
            w.writerow([i, f"{En:.9f}", f"{Ea:.9f}", f"{((En-Ea)/Ea):.6e}"])


def demo_harmonic_oscillator():
    print("\nCaso 2: Oscilador armónico (validación)")
    units = Units(hbar=1.0, m=1.0)
    omega = 5.0
    # Dominios amplios para que psi ~ 0 en bordes (≈ 4–5 sigmas)
    sigma = np.sqrt(units.hbar/(units.m*omega))
    span = 6.0 * sigma
    xmin, xmax = -span, +span
    N = 1200
    x, dx = make_grid(xmin, xmax, N)
    D2 = d2_operator_dirichlet(N, dx)
    Vx = V_harmonic(x, units.m, omega)
    H = hamiltonian_1d(x, Vx, units, D2)
    k = 6
    vals, vecs = solve_eigen(H, k=k)
    vecs = normalize_modes(vecs, dx)
    # Analítico: n = 0..k-1
    n = np.arange(0, k)
    E_anal = ho_energies_analytic(n, omega, units)
    rel_err = (vals - E_anal)/E_anal
    print("E (num)  vs  E (anal)  y error relativo:")
    for i in range(k):
        print(f"n={n[i]:d}: {vals[i]:.6f}  |  {E_anal[i]:.6f}  |  {rel_err[i]:+.2e}")
    plot_potential_and_modes(x, Vx, vals, vecs, n_show=4, title='Oscilador armónico')

    plt.savefig(os.path.join(MEMORIA_DIR, "fig_oscilador_armonico.pdf"), bbox_inches="tight")

    with open(os.path.join(EXTRAS_DIR, "oscilador_armonico_energies.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n", "E_num", "E_anal", "rel_error"])
        for i, (En, Ea) in enumerate(zip(vals, E_anal), start=0):
            w.writerow([i, f"{En:.9f}", f"{Ea:.9f}", f"{((En-Ea)/Ea):.6e}"])


def demo_finite_well():
    print("\nCaso 3: Pozo finito rectangular (estados ligados)")
    units = Units(hbar=1.0, m=1.0)
    V0 = 50.0   # profundidad
    a  = 0.5    # anchura
    x0 = 0.0
    xmin, xmax = -2.5, +2.5
    N = 1500
    x, dx = make_grid(xmin, xmax, N)
    D2 = d2_operator_dirichlet(N, dx)
    Vx = V_finite_well(x, V0=V0, a=a, x0=x0)
    H  = hamiltonian_1d(x, Vx, units, D2)
    k = 8
    vals, vecs = solve_eigen(H, k=k)
    vecs = normalize_modes(vecs, dx)
    # Filtrar ligados (E<0) en un pozo con V=-V0 dentro y 0 fuera
    mask_bound = vals < 0.0
    print(f"Estados ligados encontrados: {np.sum(mask_bound)} / {k}")
    for i, E in enumerate(vals):
        flag = "ligado" if E < 0 else "continuo"
        print(f"i={i}, E={E:.6f}  -> {flag}")
    plot_potential_and_modes(x, Vx, vals, vecs, n_show=6, title='Pozo finito rectangular')

    plt.savefig(os.path.join(MEMORIA_DIR, "fig_pozo_finito.pdf"), bbox_inches="tight")

    with open(os.path.join(EXTRAS_DIR, "pozo_finito_energies.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["i", "E", "type"])
        for i, E in enumerate(vals):
            w.writerow([i, f"{E:.9f}", "ligado" if E < 0 else "continuo"])


if __name__ == "__main__":
    demo_infinite_well()
    demo_harmonic_oscillator()
    demo_finite_well()
    plt.show()
