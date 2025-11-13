# QuantumStuff

Simulación numérica de la **ecuación de Schrödinger 1D** en Python.  
Este proyecto combina el tratamiento **estacionario** (autovalores y modos propios) y el **dependiente del tiempo** (propagación de paquetes de onda) utilizando métodos de diferencias finitas.  

>  Proyecto académico desarrollado para la asignatura *Física Cuántica I (UNED)*, extendido y documentado como laboratorio personal de simulación cuántica.

---

## Contenido principal

| Archivo | Descripción |
|----------|--------------|
| `tc1_schrodinger_1d.py` | Solución estacionaria mediante diferencias finitas. Calcula autovalores y autofunciones en pozos infinitos, finitos y osciladores armónicos. |
| `schrodinger_td.py` | Simulación temporal de paquetes de onda mediante el esquema de **Euler explícito**. Incluye los casos de onda libre, barrera de potencial y oscilador armónico. |

---

## Funcionalidades destacadas

### TC1 – Schrödinger estacionario
- Construcción matricial del **Hamiltoniano 1D** con condiciones de contorno de Dirichlet.  
- Cálculo de los **autovalores** y **autofunciones** con `scipy.sparse.linalg.eigsh`.  
- Normalización y verificación de ortogonalidad.  
- Validación con soluciones analíticas para:
  - Pozo infinito  
  - Oscilador armónico  
  - Pozo finito rectangular  

### TC2 – Schrödinger dependiente del tiempo
- Implementación de la ecuación de Schrödinger mediante **Euler explícito**.  
- Evolución de **paquetes gaussianos** bajo distintos potenciales:
  - Onda libre  
  - Barrera de potencial  
  - Oscilador armónico  
- Cálculo de probabilidades reflejadas/transmitidas.  
- Cómputo de valores esperados ⟨x⟩ y dispersión Δx.  
- Gráficas automáticas en `/memoria/figuras`.

---

## Ejemplos de resultados

### Onda libre
![onda libre](memoria/figuras/tc2_libre.pdf)

### Barrera de potencial
![barrera](memoria/figuras/tc2_barrera.pdf)

### Oscilador armónico
![oscilador](memoria/figuras/tc2_ho_xmean.pdf)

*(Si los PDF no se visualizan en GitHub, pueden abrirse directamente desde la carpeta `/memoria/figuras`.)*

---

## Aspectos numéricos

- Discretización espacial uniforme (`np.linspace`)  
- Derivadas centradas de segundo orden (operador tridiagonal)  
- Normalización en cada paso temporal  
- Estabilidad controlada por el paso `dt` y la relación de Courant  

---

## Próximas extensiones

- Integrador **Crank–Nicolson** para mayor estabilidad temporal.  
- Visualización animada interactiva (`matplotlib.animation`).  
- Implementación en **Cython** o **Numba** para benchmarking de rendimiento.  

---

> _“Quantum mechanics is not hard — it’s just complex.”_  
> — AJPM, 2025

