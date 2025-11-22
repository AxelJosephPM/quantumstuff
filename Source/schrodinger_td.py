#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
#   TC2 – Schrödinger 1D dependiente del tiempo (Euler explícito)
#   Autor: Axel (UNED – FC1)
#   Requisitos: numpy, scipy, matplotlib
# ============================================================

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.sparse as sp

# ============================================================
#  Configuración de directorios (igual que TC1)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORIA_DIR = os.path.join(BASE_DIR, "../memoria/figuras")
EXTRAS_DIR  = os.path.join(BASE_DIR, "../extras")
os.makedirs(MEMORIA_DIR, exist_ok=True)
os.makedirs(EXTRAS_DIR, exist_ok=True)

# ============================================================
#  Utilidades numéricas (reutilizando TC1)
# ============================================================

def make_grid(xmin: float, xmax: float, N: int):
    """Crea malla 1D uniforme."""
    x = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    return x, dx

def d2_operator_dirichlet(N: int, dx: float) -> sp.csr_matrix:
    """Segunda derivada centrada con Dirichlet implícitas."""
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N-1)
    D2 = sp.diags([off, main, off], [-1, 0, 1], shape=(N, N), format="csr")
    return D2 / (dx**2)

# ============================================================
#  Potenciales
# ============================================================

def V_free(x):
    return np.zeros_like(x)

def V_step(x, V0, xb):
    """Escalón: V=0 si x<xb, V=V0 si x>=xb."""
    V = np.zeros_like(x)
    V[x >= xb] = V0
    return V

def V_harmonic(x, m, omega):
    """Oscilador armónico: (1/2)mω²x²."""
    return 0.5 * m * omega**2 * x**2

# ============================================================
#  Unidades y evolución temporal
# ============================================================

@dataclass
class Units:
    hbar: float = 1.0
    m: float = 1.0

def schrodinger_step_euler(psi, Vx, dx, dt, units: Units):
    """
    Un paso temporal de Euler explícito:
    psi^{n+1} = psi^n + dt * (-i/hbar)[ -ħ²/(2m) psi'' + V psi ].
    """
    # Segunda derivada centrada
    d2psi = (np.roll(psi,-1) - 2*psi + np.roll(psi,1)) / dx**2
    
    Hpsi = -(units.hbar**2 / (2*units.m))*d2psi + Vx*psi
    psi_next = psi + dt * (-1j/units.hbar) * Hpsi
    
    # Condiciones de contorno Dirichlet
    psi_next[0]  = 0.0
    psi_next[-1] = 0.0
    
    return psi_next

def evolve(psi0, Vx, x, dt, Nt, units: Units, save_every=500):
    """Evolución temporal completa."""
    dx = x[1] - x[0]
    psi = psi0.copy()
    
    psi_list = [psi.copy()]
    t_list   = [0.0]

    for n in range(1, Nt+1):
        psi = schrodinger_step_euler(psi, Vx, dx, dt, units)
        
        # Normalización suave
        norm = np.sqrt(np.sum(np.abs(psi)**2)*dx)
        psi /= norm
        
        if n % save_every == 0:
            psi_list.append(psi.copy())
            t_list.append(n*dt)

    return np.array(psi_list), np.array(t_list)

# ============================================================
#  Valores esperados
# ============================================================

def expectation_x(x, psi, dx):
    return np.sum(x * np.abs(psi)**2) * dx

def expectation_x2(x, psi, dx):
    return np.sum((x**2) * np.abs(psi)**2) * dx

# ============================================================
#  Graficadores
# ============================================================

def plot_snapshots(x, psi_list, t_list, Vx, title, filename):
    plt.figure(figsize=(8,5))
    for psi, t in zip(psi_list, t_list):
        plt.plot(x, np.abs(psi)**2, label=f"t={t:.3f}")
    plt.plot(x, Vx/np.max(Vx+1e-12)*np.max(np.abs(psi_list[0])**2),
             '--', label="V(x) escalado")
    plt.xlabel("x")
    plt.ylabel("|Ψ|²")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MEMORIA_DIR, filename), bbox_inches="tight")
    plt.close()

# ============================================================
#  CASO 1 – Onda libre
# ============================================================

def caso_libre():
    print("=== Caso 1: Onda libre ===")

    units = Units()
    xmin, xmax, N = -10, 10, 2000
    x, dx = make_grid(xmin, xmax, N)
    Vx = V_free(x)

    sigma = 1.0
    x0    = -5.0
    k0    = 5.0
    psi0  = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2)*dx)

    dt, Tmax = 5e-6, 0.03
    Nt = int(Tmax/dt)

    psi_list, t_list = evolve(psi0, Vx, x, dt, Nt, units, save_every=2000)

    plot_snapshots(x, psi_list, t_list, Vx,
                   title="Propagación libre del paquete gaussiano",
                   filename="tc2_libre.pdf")

# ============================================================
#  CASO 2 – Barrera de potencial
# ============================================================

def caso_barrera():
    print("=== Caso 2: Barrera de potencial ===")

    units = Units()
    xmin, xmax, N = -10, 10, 2000
    x, dx = make_grid(xmin, xmax, N)

    V0 = 5.0
    xb = -3.0
    Vx = V_step(x, V0, xb)

    sigma = 1.0
    x0    = -5.0
    k0    = 6.0  # más rápido → impacto antes
    psi0  = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2)*dx)

    dt, Tmax = 5e-6, 0.5
    Nt = int(Tmax/dt)

    psi_list, t_list = evolve(psi0, Vx, x, dt, Nt, units, save_every=5000)

    plot_snapshots(x, psi_list, t_list, Vx,
                   title="Reflexión y transmisión en una barrera",
                   filename="tc2_barrera.pdf")

    # Probabilidades al final
    psi_f = psi_list[-1]
    P_R = np.sum(np.abs(psi_f[x<xb])**2)*dx
    P_T = np.sum(np.abs(psi_f[x>=xb])**2)*dx
    print(f"P_reflejada={P_R:.4f},  P_transmitida={P_T:.4f},  suma={P_R+P_T:.4f}")

# ============================================================
#  CASO 3 – Oscilador armónico
# ============================================================

def caso_oscilador():
    print("=== Caso 3: Oscilador armónico ===")

    units = Units()
    omega = 1.0
    xmin, xmax, N = -6, 6, 2000
    x, dx = make_grid(xmin, xmax, N)
    Vx = V_harmonic(x, units.m, omega)

    # Autofunciones analíticas
    def phi0(x): return np.pi**(-1/4)*np.exp(-x**2/2)
    def phi1(x): return np.sqrt(2)*x*np.pi**(-1/4)*np.exp(-x**2/2)
    def phi2(x): return (1/np.sqrt(2))*(2*x**2-1)*np.pi**(-1/4)*np.exp(-x**2/2)

    # Combinaciones exigidas
    psi0_1 = (phi0(x) + phi1(x)).astype(complex)
    psi0_1 /= np.sqrt(np.sum(np.abs(psi0_1)**2)*dx)

    psi0_2 = (phi0(x) + phi2(x)).astype(complex)
    psi0_2 /= np.sqrt(np.sum(np.abs(psi0_2)**2)*dx)

    dt, Tmax = 1e-6, 0.2
    Nt = int(Tmax/dt)

    # Evoluciones
    psi_list1, t_list1 = evolve(psi0_1, Vx, x, dt, Nt, units, save_every=4000)
    psi_list2, t_list2 = evolve(psi0_2, Vx, x, dt, Nt, units, save_every=4000)

    # Valores esperados
    X1, DX1 = [], []
    X2, DX2 = [], []
    for psi in psi_list1:
        m1 = expectation_x(x, psi, dx)
        m2 = expectation_x2(x, psi, dx)
        X1.append(m1)
        DX1.append(np.sqrt(m2 - m1**2))
    for psi in psi_list2:
        m1 = expectation_x(x, psi, dx)
        m2 = expectation_x2(x, psi, dx)
        X2.append(m1)
        DX2.append(np.sqrt(m2 - m1**2))

    # Gráficas
    plt.figure(figsize=(8,5))
    plt.plot(t_list1, X1, label="<x> (par+impar)")
    plt.plot(t_list2, X2, label="<x> (par)")
    plt.xlabel("t"); plt.ylabel("<x>")
    plt.grid(); plt.legend()
    plt.title("Evolución del valor esperado en el HO")
    plt.savefig(os.path.join(MEMORIA_DIR,"tc2_ho_xmean.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(t_list1, DX1, label="Δx (par+impar)")
    plt.plot(t_list2, DX2, label="Δx (par)")
    plt.xlabel("t"); plt.ylabel("Δx")
    plt.grid(); plt.legend()
    plt.title("Dispersión en el HO")
    plt.savefig(os.path.join(MEMORIA_DIR,"tc2_ho_delta_x.pdf"), bbox_inches="tight")
    plt.close()

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    caso_libre()
    caso_barrera()
    caso_oscilador()
    plt.show()
