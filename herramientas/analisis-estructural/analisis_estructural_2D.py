# -*- coding: utf-8 -*-
"""
ANALISIS ESTRUCTURAL DE PORTICOS PLANOS (2D)
=============================================
Estatico lineal | Modal | Modal espectral (CQC/SRSS) | No lineal (pushover)

Como usar en Spyder:
  1. Abra este archivo en Spyder (Anaconda incluye numpy, scipy, pandas,
     matplotlib y openpyxl, que son las unicas dependencias).
  2. Ejecute el script completo con F5, o celda por celda con Ctrl+Enter
     (las celdas estan delimitadas con "# %%").
  3. Modifique la CELDA 2 (geometria), CELDA 3 (materiales/secciones/
     capacidades) y CELDA 4 (apoyos, masas y cargas) para su propio modelo.
     El resto del script (motor de elementos finitos, analisis y graficos)
     no requiere cambios para correr el ejemplo incluido.

Unidades consistentes utilizadas en todo el script: kN, m, kN/m2, kN*s2/m
(masa = peso/9.81). Si desea usar otro sistema de unidades, sea consistente
en todas las CELDAS de entrada.

Alcance y limitaciones (leer antes de usar en un proyecto real):
  - Elemento tipo portico 2D (viga-columna de Euler-Bernoulli, 3 GDL/nodo).
  - El analisis no lineal usa rotulas plasticas concentradas
    elasto-perfectamente-plasticas (sin endurecimiento, sin interaccion
    N-M-V, sin degradacion ciclica) mediante el metodo "evento a evento".
    Es una idealizacion simplificada con fines didacticos/de prediseno.
    Para diseno final use software especializado (ETABS, SAP2000, OpenSees)
    con curvas de capacidad reales de las secciones (momento-rotacion,
    interaccion P-M-M, efectos ciclicos, etc.).
  - Las cargas distribuidas solo se aplican en la direccion transversal
    local (tipico para vigas horizontales bajo carga de gravedad).
  - El espectro de diseno incluido tiene la forma del espectro elastico
    de la NSR-10 (Colombia, Fig. A.2.6-1), generico y parametrizable
    (Aa, Av, Fa, Fv, I, R); reemplacelo si su proyecto usa otra norma.
"""

# %% CELDA 0: LIBRERIAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

G = 9.81  # gravedad, m/s2, para convertir peso (kN) a masa (kN*s2/m)


# %% CELDA 1: FUNCIONES DEL ELEMENTO FINITO (motor de calculo)
# No es necesario modificar esta celda para usar el script.

def k_local(E, A, I, L, hinge_i=False, hinge_j=False):
    """Matriz de rigidez local (6x6) de un elemento portico 2D (Euler-Bernoulli).
    dof local = [u_i, v_i, theta_i, u_j, v_j, theta_j].
    hinge_i / hinge_j: si es True, se libera el momento en ese extremo
    (rotula), condensando estaticamente el GDL de rotacion correspondiente.
    """
    k = np.array([
        [ E*A/L,      0,            0,        -E*A/L,      0,            0        ],
        [ 0,        12*E*I/L**3,  6*E*I/L**2,  0,       -12*E*I/L**3,  6*E*I/L**2 ],
        [ 0,         6*E*I/L**2,  4*E*I/L,     0,        -6*E*I/L**2,  2*E*I/L    ],
        [-E*A/L,      0,            0,         E*A/L,      0,            0        ],
        [ 0,       -12*E*I/L**3, -6*E*I/L**2,  0,        12*E*I/L**3, -6*E*I/L**2 ],
        [ 0,         6*E*I/L**2,  2*E*I/L,     0,        -6*E*I/L**2,  4*E*I/L    ],
    ])
    if hinge_i and hinge_j:
        kk = np.zeros((6, 6))
        kk[0, 0] = E*A/L; kk[0, 3] = -E*A/L
        kk[3, 0] = -E*A/L; kk[3, 3] = E*A/L
        return kk
    if hinge_i or hinge_j:
        c = 2 if hinge_i else 5
        keep = [i for i in range(6) if i != c]
        kcc = k[np.ix_(keep, keep)] - np.outer(k[keep, c], k[c, keep])/k[c, c]
        kk = np.zeros((6, 6))
        kk[np.ix_(keep, keep)] = kcc
        return kk
    return k


def T_matrix(cx, cy):
    """Matriz de transformacion local-global (6x6) a partir de los cosenos
    directores (cx, cy) del eje local x del elemento."""
    R = np.array([[cx, cy, 0], [-cy, cx, 0], [0, 0, 1]])
    T = np.zeros((6, 6))
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    return T


def fem_udl(w, L):
    """Vector de fuerzas de empotramiento (fixed-end forces) para una carga
    uniformemente distribuida w (kN/m, signo + = eje local +y) sobre un
    elemento de longitud L. Formula clasica, validada contra la solucion
    cerrada de una viga empotrada-empotrada (momentos wL^2/12, wL^2/24)."""
    return np.array([0, w*L/2, w*L**2/12, 0, w*L/2, -w*L**2/12])


class Modelo:
    """Contenedor del modelo estructural: nodos, elementos, apoyos, masas
    y cargas. Ensambla las matrices globales y resuelve los distintos tipos
    de analisis."""

    def __init__(self):
        self.nodes = {}        # id -> (x, y)
        self.restraints = {}   # id -> (rx, ry, rz) bool
        self.node_mass = {}    # id -> masa traslacional (kN*s2/m)
        self.elements = []     # lista de dicts

    def agregar_nodo(self, nid, x, y, rx=False, ry=False, rz=False, masa=0.0):
        self.nodes[nid] = (x, y)
        self.restraints[nid] = (rx, ry, rz)
        self.node_mass[nid] = masa

    def agregar_elemento(self, eid, ni, nj, E, A, I, w=0.0, Mp_i=None, Mp_j=None, tipo="elemento"):
        self.elements.append(dict(id=eid, ni=ni, nj=nj, E=E, A=A, I=I, w=w,
                                   Mp_i=Mp_i, Mp_j=Mp_j, tipo=tipo,
                                   hinge_i=False, hinge_j=False))

    # ---- infraestructura de indices / GDL ----
    def _preparar(self):
        self.node_ids = sorted(self.nodes.keys())
        self.idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.ndof = 3*len(self.node_ids)
        for e in self.elements:
            xi, yi = self.nodes[e['ni']]
            xj, yj = self.nodes[e['nj']]
            e['L'] = float(np.hypot(xj-xi, yj-yi))
            e['cx'] = (xj-xi)/e['L']
            e['cy'] = (yj-yi)/e['L']
            dofi = [3*self.idx[e['ni']]+k for k in range(3)]
            dofj = [3*self.idx[e['nj']]+k for k in range(3)]
            e['dofs'] = dofi+dofj

    def dof(self, nid, comp):
        """comp: 0=ux, 1=uy, 2=rz"""
        return 3*self.idx[nid]+comp

    def _restrained_dofs(self):
        rdofs = []
        for nid in self.node_ids:
            rx, ry, rz = self.restraints[nid]
            base = 3*self.idx[nid]
            if rx: rdofs.append(base+0)
            if ry: rdofs.append(base+1)
            if rz: rdofs.append(base+2)
        return rdofs

    def ensamblar_K(self, usar_rotulas=False):
        K = np.zeros((self.ndof, self.ndof))
        for e in self.elements:
            hi = e['hinge_i'] if usar_rotulas else False
            hj = e['hinge_j'] if usar_rotulas else False
            k = k_local(e['E'], e['A'], e['I'], e['L'], hi, hj)
            T = T_matrix(e['cx'], e['cy'])
            kg = T.T @ k @ T
            d = e['dofs']
            K[np.ix_(d, d)] += kg
        return K

    def vector_cargas_nodales(self, cargas_nodales):
        """cargas_nodales: dict nid -> (Fx, Fy, Mz)"""
        F = np.zeros(self.ndof)
        for nid, (fx, fy, mz) in cargas_nodales.items():
            F[self.dof(nid, 0)] += fx
            F[self.dof(nid, 1)] += fy
            F[self.dof(nid, 2)] += mz
        return F

    def vector_cargas_distribuidas(self):
        """Fuerzas nodales equivalentes (consistentes) de las cargas w de
        cada elemento. Se agregan directamente (mismo signo) al vector F."""
        F = np.zeros(self.ndof)
        for e in self.elements:
            if e['w'] == 0:
                continue
            f0 = fem_udl(e['w'], e['L'])
            T = T_matrix(e['cx'], e['cy'])
            F[e['dofs']] += T.T @ f0
        return F

    def matriz_masa_diagonal(self):
        M = np.zeros(self.ndof)
        for nid, m in self.node_mass.items():
            M[self.dof(nid, 0)] = m
            M[self.dof(nid, 1)] = m
        return M

    def resolver_lineal(self, F):
        """Resuelve K u = F con las condiciones de apoyo definidas.
        Retorna (u, reacciones) ambos vectores de tamano ndof."""
        K = self.ensamblar_K(usar_rotulas=False)
        rdofs = self._restrained_dofs()
        free = [i for i in range(self.ndof) if i not in rdofs]
        Kff = K[np.ix_(free, free)]
        Ff = F[free]
        uf = np.linalg.solve(Kff, Ff)
        u = np.zeros(self.ndof)
        u[free] = uf
        reacciones = K @ u - F
        return u, reacciones

    def fuerzas_por_elemento(self, u, incluir_carga_distribuida=True):
        """Retorna dict eid -> [Ni,Vi,Mi,Nj,Vj,Mj] (fuerzas locales).

        incluir_carga_distribuida: dejar en True para desplazamientos que
        provienen de resolver K u = F (analisis estatico), donde la carga
        distribuida w SI actua junto con el desplazamiento u y por lo tanto
        debe sumarse su efecto (correccion de empotramiento). Debe ponerse
        en False para desplazamientos modales/dinamicos (u = Gamma*phi*Sd),
        que no tienen una carga distribuida fisica asociada -- alli la
        fuerza interna es simplemente la rigidez elastica por el
        desplazamiento modal, sin ninguna correccion de carga."""
        out = {}
        for e in self.elements:
            T = T_matrix(e['cx'], e['cy'])
            k = k_local(e['E'], e['A'], e['I'], e['L'])
            ul = T @ u[e['dofs']]
            if incluir_carga_distribuida and e['w'] != 0:
                f0 = fem_udl(e['w'], e['L'])
            else:
                f0 = np.zeros(6)
            out[e['id']] = k @ ul - f0
        return out

    def diagrama_elemento(self, e, f_local, n=41):
        """Devuelve x, N(x), V(x), M(x) a lo largo del elemento (formulas
        validadas contra la solucion cerrada de viga empotrada-empotrada
        bajo carga uniforme y contra el equilibrio global de un portico)."""
        Ni, Vi, Mi, Nj, Vj, Mj = f_local
        w = e['w']
        x = np.linspace(0, e['L'], n)
        N = -Ni*np.ones_like(x)
        V = Vi + w*x
        M = -Mi + Vi*x + 0.5*w*x**2
        return x, N, V, M


# %% CELDA 2: GEOMETRIA DEL EJEMPLO (portico de 3 niveles y 2 luces)
# Reemplace esta celda con la geometria de su propio proyecto.

modelo = Modelo()

Hpiso = 3.0     # altura de entrepiso, m
Lvano = 5.0     # luz de los vanos, m
niveles = [0, 1, 2, 3]          # 0 = base
columnas_x = [0.0, Lvano, 2*Lvano]

nid_por_nivel = {}
nid = 1
for niv in niveles:
    y = niv*Hpiso
    fila = []
    for x in columnas_x:
        fijo = (niv == 0)
        modelo.agregar_nodo(nid, x, y, rx=fijo, ry=fijo, rz=fijo)
        fila.append(nid)
        nid += 1
    nid_por_nivel[niv] = fila


# %% CELDA 3: MATERIALES, SECCIONES Y CAPACIDADES A FLEXION (Mp)
# fc = 21 MPa (2.1e4 kN/m2), formula NSR-10/ACI: Ec = 4700*sqrt(fc' en MPa)
fc = 21.0                                   # MPa
Ec = 4700*np.sqrt(fc)*1000.0                # kN/m2 (4700*sqrt(21)=21538 MPa)

# Columnas 0.40x0.40 m, vigas 0.30x0.45 m
b_col, h_col = 0.40, 0.40
b_vig, h_vig = 0.30, 0.45
A_col, I_col = b_col*h_col, b_col*h_col**3/12
A_vig, I_vig = b_vig*h_vig, b_vig*h_vig**3/12

# Momentos plasticos (capacidad a flexion) -- valores representativos.
# EN UN PROYECTO REAL reemplace estos valores por la capacidad Mn de cada
# seccion, calculada con el area de acero, fc y fy reales (ACI 318 / NSR-10).
Mp_columna = 180.0   # kN-m
Mp_viga = 140.0      # kN-m

eid = 1
for niv in [1, 2, 3]:
    inferior = nid_por_nivel[niv-1]
    superior = nid_por_nivel[niv]
    for k in range(3):
        modelo.agregar_elemento(eid, inferior[k], superior[k], Ec, A_col, I_col,
                                 Mp_i=Mp_columna, Mp_j=Mp_columna, tipo=f"Columna N{niv}")
        eid += 1
    for k in range(2):
        modelo.agregar_elemento(eid, superior[k], superior[k+1], Ec, A_vig, I_vig,
                                 Mp_i=Mp_viga, Mp_j=Mp_viga, tipo=f"Viga N{niv}")
        eid += 1

modelo._preparar()


# %% CELDA 4: APOYOS (ya definidos en celda 2), MASAS Y CARGAS

# --- Cargas de gravedad sobre las vigas (carga muerta + viva de servicio) ---
w_gravedad = -22.0   # kN/m (signo negativo = hacia abajo, eje global -y)
for e in modelo.elements:
    if e['tipo'].startswith("Viga"):
        e['w'] = w_gravedad

# --- Masas concentradas por nivel (para el analisis modal) ---
# Peso sismico por nivel (ejemplo): W = 320 kN en niveles 1-2, 260 kN en la
# cubierta (nivel 3), repartido por igual entre los 3 nodos de cada nivel.
peso_por_nivel = {1: 320.0, 2: 320.0, 3: 260.0}
for niv, W in peso_por_nivel.items():
    m_nodo = (W/G)/3
    for nid_ in nid_por_nivel[niv]:
        modelo.node_mass[nid_] = m_nodo

# --- Cargas nodales laterales de ejemplo (p.ej. viento o sismo estatico) ---
cargas_nodales_estatico = {}
Flat = {1: 15.0, 2: 25.0, 3: 20.0}   # kN por nivel, repartidas en sus 3 nodos
for niv, Ft in Flat.items():
    for nid_ in nid_por_nivel[niv]:
        cargas_nodales_estatico[nid_] = (Ft/3, 0.0, 0.0)

print("Modelo definido:", len(modelo.nodes), "nodos,", len(modelo.elements), "elementos.")


# %% CELDA 5: ANALISIS ESTATICO LINEAL

F_estatico = (modelo.vector_cargas_distribuidas()
              + modelo.vector_cargas_nodales(cargas_nodales_estatico))
u_estatico, reacciones_estatico = modelo.resolver_lineal(F_estatico)
fuerzas_estatico = modelo.fuerzas_por_elemento(u_estatico)

# Chequeo de equilibrio global (verificacion de consistencia del modelo)
rdofs = modelo._restrained_dofs()
Rx = sum(reacciones_estatico[d] for d in rdofs if d % 3 == 0)
Ry = sum(reacciones_estatico[d] for d in rdofs if d % 3 == 1)
Fx_aplicado = sum(F_estatico[d] for d in range(modelo.ndof) if d % 3 == 0)
Fy_aplicado = sum(F_estatico[d] for d in range(modelo.ndof) if d % 3 == 1)
print("=== Analisis estatico lineal ===")
print(f"Equilibrio Fx: aplicado={Fx_aplicado:.3f} kN | reacciones={-Rx:.3f} kN")
print(f"Equilibrio Fy: aplicado={Fy_aplicado:.3f} kN | reacciones={-Ry:.3f} kN")


# %% CELDA 6: TABLA DE RESULTADOS ESTATICOS POR ELEMENTO

filas = []
for e in modelo.elements:
    f = fuerzas_estatico[e['id']]
    filas.append(dict(Elemento=e['id'], Tipo=e['tipo'], Nodo_i=e['ni'], Nodo_j=e['nj'],
                       L_m=round(e['L'], 2),
                       N_i_kN=f[0], V_i_kN=f[1], M_i_kNm=f[2],
                       N_j_kN=f[3], V_j_kN=f[4], M_j_kNm=f[5]))
tabla_estatico = pd.DataFrame(filas).set_index("Elemento")
print("\nTabla de fuerzas internas por elemento (analisis estatico):")
print(tabla_estatico)

tabla_reacciones = pd.DataFrame([
    dict(Nodo=nid, Rx_kN=reacciones_estatico[modelo.dof(nid, 0)],
         Ry_kN=reacciones_estatico[modelo.dof(nid, 1)],
         Mz_kNm=reacciones_estatico[modelo.dof(nid, 2)])
    for nid in modelo.node_ids if any(modelo.restraints[nid])
]).set_index("Nodo")
print("\nReacciones en los apoyos:")
print(tabla_reacciones)


# %% CELDA 7: GRAFICOS DEL ANALISIS ESTATICO (estructura, deformada, diagramas)

def graficar_estructura(ax, modelo, u=None, escala=1.0, color="0.75", color_def="C0",
                         etiquetas=False):
    for e in modelo.elements:
        xi, yi = modelo.nodes[e['ni']]
        xj, yj = modelo.nodes[e['nj']]
        ax.plot([xi, xj], [yi, yj], color=color, lw=2, zorder=1)
        if u is not None:
            xi_d = xi + escala*u[modelo.dof(e['ni'], 0)]
            yi_d = yi + escala*u[modelo.dof(e['ni'], 1)]
            xj_d = xj + escala*u[modelo.dof(e['nj'], 0)]
            yj_d = yj + escala*u[modelo.dof(e['nj'], 1)]
            ax.plot([xi_d, xj_d], [yi_d, yj_d], color=color_def, lw=2, zorder=2)
    xs = [p[0] for p in modelo.nodes.values()]
    ys = [p[1] for p in modelo.nodes.values()]
    if etiquetas:
        for nid, (x, y) in modelo.nodes.items():
            ax.annotate(str(nid), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_aspect("equal")
    ax.margins(0.15)


fig, ax = plt.subplots(figsize=(7, 6))
escala_def = 200
graficar_estructura(ax, modelo, u=u_estatico, escala=escala_def, etiquetas=True)
ax.set_title(f"Estructura original y deformada (analisis estatico, escala x{escala_def})")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout()
fig.savefig("resultado_01_deformada_estatica.png", dpi=150)

# Diagramas N-V-M por elemento (todas las vigas y columnas en una figura por tipo)
def graficar_diagramas(modelo, fuerzas, titulo_extra=""):
    tipos = sorted(set(e['tipo'] for e in modelo.elements))
    for tipo in tipos:
        elems = [e for e in modelo.elements if e['tipo'] == tipo]
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
        for e in elems:
            f = fuerzas[e['id']]
            x, N, V, M = modelo.diagrama_elemento(e, f)
            axs[0].plot(x, N, label=f"Elem {e['id']}")
            axs[1].plot(x, V, label=f"Elem {e['id']}")
            axs[2].plot(x, M, label=f"Elem {e['id']}")
        for ax_, titulo in zip(axs, ["Axial N (kN)", "Cortante V (kN)", "Momento M (kN-m)"]):
            ax_.axhline(0, color="k", lw=0.6)
            ax_.set_title(titulo, fontsize=10)
            ax_.set_xlabel("x (m)")
            ax_.grid(alpha=0.3)
        axs[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"Diagramas {tipo}{titulo_extra}")
        fig.tight_layout()
        nombre = f"resultado_02_diagramas_{tipo.replace(' ', '_')}{titulo_extra.replace(' ', '_')}.png"
        fig.savefig(nombre, dpi=150)

graficar_diagramas(modelo, fuerzas_estatico, titulo_extra=" (estatico)")


# %% CELDA 8: ANALISIS MODAL
# Condensacion estatica de los GDL sin masa (rotaciones) -- metodo estandar
# en dinamica de estructuras (Chopra) cuando solo se usan masas traslacionales
# concentradas por piso.

def analisis_modal(modelo, n_modos=None):
    K = modelo.ensamblar_K(usar_rotulas=False)
    mass_vec = modelo.matriz_masa_diagonal()
    rdofs = modelo._restrained_dofs()
    free = [i for i in range(modelo.ndof) if i not in rdofs]
    Kff = K[np.ix_(free, free)]
    mass_free = mass_vec[free]

    dyn = [i for i, m in enumerate(mass_free) if m > 0]
    cond = [i for i, m in enumerate(mass_free) if m == 0]

    Ktt = Kff[np.ix_(dyn, dyn)]
    Kto = Kff[np.ix_(dyn, cond)]
    Kot = Kff[np.ix_(cond, dyn)]
    Koo = Kff[np.ix_(cond, cond)]
    Kcond = Ktt - Kto @ np.linalg.solve(Koo, Kot)
    Mtt = np.diag(mass_free[dyn])

    vals, vecs = eigh(Kcond, Mtt)
    vals = np.clip(vals, 0, None)
    omegas = np.sqrt(vals)
    orden = np.argsort(omegas)
    omegas = omegas[orden]
    vecs = vecs[:, orden]
    if n_modos is not None:
        omegas = omegas[:n_modos]
        vecs = vecs[:, :n_modos]

    # recuperar el vector modal completo (incluyendo rotaciones condensadas)
    n_modos_ = vecs.shape[1]
    phi_o = -np.linalg.solve(Koo, Kot) @ vecs
    phi_free = np.zeros((len(free), n_modos_))
    for j, i in enumerate(dyn):
        phi_free[i, :] = vecs[j, :]
    for j, i in enumerate(cond):
        phi_free[i, :] = phi_o[j, :]
    phi_full = np.zeros((modelo.ndof, n_modos_))
    phi_full[free, :] = phi_free

    return dict(omegas=omegas, vecs_dinamicos=vecs, Mtt=Mtt, dyn=dyn, free=free,
                phi_full=phi_full)


resultado_modal = analisis_modal(modelo)
omegas = resultado_modal['omegas']
periodos = 2*np.pi/omegas
frecuencias = omegas/(2*np.pi)

# masas efectivas (direccion horizontal global x, tipico para sismo)
free = resultado_modal['free']
dyn = resultado_modal['dyn']
Mtt = resultado_modal['Mtt']
vecs = resultado_modal['vecs_dinamicos']
kind_libre = [d % 3 for d in free]
kind_dyn = [kind_libre[i] for i in dyn]
iota = np.array([1.0 if k == 0 else 0.0 for k in kind_dyn])
Ln = vecs.T @ Mtt @ iota
Mn_star = np.array([vecs[:, i] @ Mtt @ vecs[:, i] for i in range(len(omegas))])
Gamma_n = Ln/Mn_star
masa_efectiva = Ln**2/Mn_star
masa_total_x = iota @ Mtt @ iota
masa_efectiva_pct = 100*masa_efectiva/masa_total_x

tabla_modal = pd.DataFrame(dict(
    Modo=np.arange(1, len(omegas)+1),
    T_s=periodos, f_Hz=frecuencias, omega_rad_s=omegas,
    Masa_efectiva_pct=masa_efectiva_pct,
    Masa_efectiva_acum_pct=np.cumsum(masa_efectiva_pct),
)).set_index("Modo")

print("\n=== Analisis modal ===")
print(f"Masa total participante en direccion X: {masa_total_x:.3f} kN*s2/m")
print(tabla_modal.round(4))


# %% CELDA 9: GRAFICOS DEL ANALISIS MODAL (formas modales)

def graficar_modo(modelo, phi_full, n, escala=2.0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    graficar_estructura(ax, modelo, u=phi_full[:, n], escala=escala)
    ax.set_title(f"Modo {n+1}  (T = {periodos[n]:.3f} s)")
    return ax


n_modos_graficar = min(3, len(omegas))
fig, axs = plt.subplots(1, n_modos_graficar, figsize=(5*n_modos_graficar, 5))
if n_modos_graficar == 1:
    axs = [axs]
for i in range(n_modos_graficar):
    graficar_modo(modelo, resultado_modal['phi_full'], i, ax=axs[i])
fig.suptitle("Formas modales (escala grafica arbitraria)")
fig.tight_layout()
fig.savefig("resultado_03_formas_modales.png", dpi=150)


# %% CELDA 10: ESPECTRO DE DISENO Y ANALISIS MODAL ESPECTRAL (CQC / SRSS)
# Espectro elastico con la forma de la NSR-10 (Colombia), Fig. A.2.6-1.
# Reemplace por el espectro de su norma/proyecto si es diferente.

Aa, Av = 0.15, 0.20     # aceleracion pico efectiva (ejemplo, ajustar a su zona)
Fa, Fv = 1.2, 1.75      # coeficientes de sitio (ejemplo, suelo tipo D)
I_importancia = 1.0     # coeficiente de importancia
R_reduccion = 1.0       # 1.0 = espectro elastico (sin reducir); use R real para fuerzas de diseno
zeta = 0.05             # razon de amortiguamiento critico (5%, tipico en concreto)


def espectro_nsr10(T, Aa=Aa, Av=Av, Fa=Fa, Fv=Fv, I=I_importancia, R=R_reduccion):
    """Sa(T) en unidades de g. Forma del espectro elastico de diseno NSR-10.
    Acepta un escalar o un arreglo; siempre retorna un arreglo numpy con la
    misma forma que la entrada (usar float(...) o [0] en el llamador si se
    necesita un escalar)."""
    escalar = np.isscalar(T) or np.ndim(T) == 0
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Tc = 0.48*Av*Fv/(Aa*Fa)
    T0 = 0.1*Tc
    TL = 2.4*Fv
    Sa = np.empty_like(T)
    m0 = T < T0
    m1 = (T >= T0) & (T <= Tc)
    m2 = (T > Tc) & (T <= TL)
    m3 = T > TL
    Sa[m0] = 2.5*Aa*Fa*(0.4 + 0.6*T[m0]/T0) if T0 > 0 else 2.5*Aa*Fa
    Sa[m1] = 2.5*Aa*Fa
    Sa[m2] = 1.2*Av*Fv/T[m2]
    Sa[m3] = 1.2*Av*Fv*TL/T[m3]**2
    Sa = Sa*I/R
    return float(Sa[0]) if escalar else Sa


def rho_cqc(zeta, wi, wj):
    r = wj/wi
    return (8*zeta**2*(1+r)*r**1.5) / ((1-r**2)**2 + 4*zeta**2*r*(1+r)**2)


def analisis_espectral(modelo, resultado_modal, espectro_fn, zeta=0.05, metodo="CQC"):
    omegas = resultado_modal['omegas']
    phi_full = resultado_modal['phi_full']
    n_modos = len(omegas)
    T = 2*np.pi/omegas
    Sa_g = espectro_fn(T)              # en g
    Sa = Sa_g*G                        # m/s2

    free = resultado_modal['free']; dyn = resultado_modal['dyn']
    Mtt = resultado_modal['Mtt']; vecs = resultado_modal['vecs_dinamicos']
    kind_libre = [d % 3 for d in free]
    kind_dyn = [kind_libre[i] for i in dyn]
    iota = np.array([1.0 if k == 0 else 0.0 for k in kind_dyn])
    Ln = vecs.T @ Mtt @ iota
    Mn_star = np.array([vecs[:, i] @ Mtt @ vecs[:, i] for i in range(n_modos)])
    Gamma_n = Ln/Mn_star

    # desplazamiento modal maximo (pseudo-espectral) y fuerzas por elemento
    Sd = Sa/omegas**2                                  # m
    u_modal = Gamma_n[np.newaxis, :]*phi_full*Sd[np.newaxis, :]   # ndof x n_modos

    fuerzas_modal = np.zeros((len(modelo.elements), 6, n_modos))
    for jm in range(n_modos):
        fmod = modelo.fuerzas_por_elemento(u_modal[:, jm], incluir_carga_distribuida=False)
        for ie, e in enumerate(modelo.elements):
            fuerzas_modal[ie, :, jm] = fmod[e['id']]

    if metodo.upper() == "CQC":
        rho = np.ones((n_modos, n_modos))
        for i in range(n_modos):
            for j in range(n_modos):
                if i != j:
                    rho[i, j] = rho_cqc(zeta, omegas[i], omegas[j])
        combinado = np.zeros((len(modelo.elements), 6))
        for ie in range(len(modelo.elements)):
            for c in range(6):
                v = fuerzas_modal[ie, c, :]
                combinado[ie, c] = np.sqrt(max(v @ rho @ v, 0.0))
    else:  # SRSS
        combinado = np.sqrt(np.sum(fuerzas_modal**2, axis=2))

    return dict(T=T, Sa_g=Sa_g, Gamma_n=Gamma_n, u_modal=u_modal,
                fuerzas_por_modo=fuerzas_modal, fuerzas_combinadas=combinado)


resultado_espectral = analisis_espectral(modelo, resultado_modal, espectro_nsr10,
                                          zeta=zeta, metodo="CQC")

filas = []
for ie, e in enumerate(modelo.elements):
    c = resultado_espectral['fuerzas_combinadas'][ie]
    filas.append(dict(Elemento=e['id'], Tipo=e['tipo'],
                       N_env_kN=c[0], V_env_kN=max(c[1], c[4]),
                       M_env_kNm=max(c[2], c[5])))
tabla_espectral = pd.DataFrame(filas).set_index("Elemento")
print("\n=== Analisis modal espectral (envolvente CQC, valores absolutos) ===")
print(tabla_espectral)


# %% CELDA 11: GRAFICOS DEL ANALISIS ESPECTRAL

fig, ax = plt.subplots(figsize=(6, 4.5))
Tplot = np.linspace(0.001, 3.0, 400)
ax.plot(Tplot, espectro_nsr10(Tplot), color="0.4", lw=1.5, label="Espectro de diseno")
ax.scatter(resultado_espectral['T'], espectro_nsr10(resultado_espectral['T']),
           color="C3", zorder=5, label="Periodos modales")
for i, Tm in enumerate(resultado_espectral['T'][:5]):
    ax.annotate(f"Modo {i+1}", (Tm, espectro_nsr10(np.array([Tm]))[0]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("T (s)"); ax.set_ylabel("Sa (g)")
ax.set_title("Espectro elastico de diseno")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig("resultado_04_espectro.png", dpi=150)

def graficar_envolvente_espectral(modelo, combinado):
    """Grafica de barras de la envolvente CQC/SRSS por elemento (los valores
    combinados son siempre positivos por construccion -- son una envolvente,
    no un estado de equilibrio unico -- por lo que se muestran como barras
    en los extremos i/j y no como un diagrama continuo N-V-M)."""
    tipos = sorted(set(e['tipo'] for e in modelo.elements))
    for tipo in tipos:
        idxs = [ie for ie, e in enumerate(modelo.elements) if e['tipo'] == tipo]
        ids = [modelo.elements[ie]['id'] for ie in idxs]
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
        x = np.arange(len(ids))
        w_ = 0.35
        axs[0].bar(x-w_/2, [combinado[ie, 1] for ie in idxs], w_, label="Extremo i")
        axs[0].bar(x+w_/2, [combinado[ie, 4] for ie in idxs], w_, label="Extremo j")
        axs[0].set_title("Cortante envolvente |V| (kN)", fontsize=10)
        axs[1].bar(x-w_/2, [combinado[ie, 2] for ie in idxs], w_, label="Extremo i")
        axs[1].bar(x+w_/2, [combinado[ie, 5] for ie in idxs], w_, label="Extremo j")
        axs[1].set_title("Momento envolvente |M| (kN-m)", fontsize=10)
        for ax_ in axs:
            ax_.set_xticks(x); ax_.set_xticklabels(ids)
            ax_.set_xlabel("Elemento"); ax_.grid(alpha=0.3, axis="y")
            ax_.legend(fontsize=8)
        fig.suptitle(f"Envolvente espectral CQC -- {tipo}")
        fig.tight_layout()
        fig.savefig(f"resultado_05_envolvente_{tipo.replace(' ', '_')}.png", dpi=150)


graficar_envolvente_espectral(modelo, resultado_espectral['fuerzas_combinadas'])


# %% CELDA 12: ANALISIS NO LINEAL (PUSHOVER -- ROTULAS PLASTICAS)
# Metodo "evento a evento": se incrementa el patron de carga lateral hasta
# que el momento en algun extremo de elemento alcanza su capacidad Mp; en
# ese instante se inserta una rotula (condensacion estatica del GDL de
# rotacion) y se continua. Idealizacion elasto-perfectamente-plastica.

def patron_carga_lateral(modelo, nid_por_nivel, resultado_modal=None):
    """Patron lateral proporcional a masa*altura (aproximacion habitual del
    modo fundamental para pushover), normalizado a un cortante basal unitario."""
    F = np.zeros(modelo.ndof)
    pesos = []
    for niv, nids in nid_por_nivel.items():
        if niv == 0:
            continue
        y = modelo.nodes[nids[0]][1]
        m_total = sum(modelo.node_mass[n] for n in nids)
        pesos.append((niv, nids, m_total*y))
    suma = sum(p[2] for p in pesos)
    for niv, nids, peso in pesos:
        Fi = peso/suma
        for n in nids:
            F[modelo.dof(n, 0)] += Fi/len(nids)
    return F


def pushover(modelo, F_pattern, nodo_control, dof_control=0, du_lambda=0.02,
             lambda_max=500.0, max_pasos=20000, cond_max=1e12):
    for e in modelo.elements:
        e['hinge_i'] = False
        e['hinge_j'] = False
        e['force_cum'] = np.zeros(6)

    rdofs = modelo._restrained_dofs()
    free = [i for i in range(modelo.ndof) if i not in rdofs]
    u_cum = np.zeros(modelo.ndof)
    lam_cum = 0.0
    hist_lambda, hist_u = [0.0], [0.0]
    hist_hinges = []
    ctrl_dof = modelo.dof(nodo_control, dof_control)

    for paso in range(max_pasos):
        K = modelo.ensamblar_K(usar_rotulas=True)
        Kff = K[np.ix_(free, free)]
        if np.linalg.cond(Kff) > cond_max:
            print(f"  Mecanismo (matriz mal condicionada) tras {len(hist_hinges)} rotulas, "
                  f"lambda={lam_cum:.3f}")
            break

        dF = du_lambda*F_pattern[free]
        d_du = np.linalg.solve(Kff, dF)
        du_full = np.zeros(modelo.ndof)
        du_full[free] = d_du

        # buscar el primer elemento/extremo que alcanza Mp en este incremento
        escala = 1.0
        critico = None
        incrementos = {}
        for e in modelo.elements:
            T = T_matrix(e['cx'], e['cy'])
            k = k_local(e['E'], e['A'], e['I'], e['L'], e['hinge_i'], e['hinge_j'])
            du_local = T @ du_full[e['dofs']]
            df = k @ du_local
            incrementos[e['id']] = df
            for extremo, im, hk, Mp in [('i', 2, 'hinge_i', e['Mp_i']),
                                         ('j', 5, 'hinge_j', e['Mp_j'])]:
                if e[hk] or Mp is None:
                    continue
                M_now = e['force_cum'][im]
                dM = df[im]
                if abs(M_now) >= Mp - 1e-9 or dM == 0:
                    continue
                for signo in (1, -1):
                    s = (signo*Mp - M_now)/dM
                    if 0 < s <= 1 and (critico is None or s < critico[0]):
                        critico = (s, e['id'], hk, signo*Mp)
        if critico is not None:
            escala = critico[0]

        lam_cum += du_lambda*escala
        u_cum += du_full*escala
        for e in modelo.elements:
            e['force_cum'] += incrementos[e['id']]*escala
        hist_lambda.append(lam_cum)
        hist_u.append(u_cum[ctrl_dof])

        if critico is not None:
            _, eid, hk, Mval = critico
            for e in modelo.elements:
                if e['id'] == eid:
                    e[hk] = True
                    hist_hinges.append(dict(orden=len(hist_hinges)+1, elemento=eid,
                                             extremo=hk.split('_')[1],
                                             lambda_=lam_cum, M_kNm=Mval))
                    print(f"  Rotula #{len(hist_hinges)}: elemento {eid}, extremo "
                          f"{hk.split('_')[1]}, cortante basal={lam_cum:.3f} kN, M={Mval:.2f} kN-m")
                    break
            if all(e['hinge_i'] and e['hinge_j'] for e in modelo.elements if e['Mp_i'] is not None):
                break
        if lam_cum > lambda_max:
            break

    return dict(lam=np.array(hist_lambda), u_control=np.array(hist_u),
                rotulas=pd.DataFrame(hist_hinges))


# Se usa el mismo patron lateral del analisis estatico, escalado; el nodo de
# control es el nodo central del ultimo nivel (techo).
F_pushover = patron_carga_lateral(modelo, nid_por_nivel)
nodo_control = nid_por_nivel[max(nid_por_nivel)][1]

print("\n=== Analisis no lineal (pushover) ===")
resultado_pushover = pushover(modelo, F_pushover, nodo_control)

tabla_rotulas = resultado_pushover['rotulas']
print("\nSecuencia de formacion de rotulas:")
print(tabla_rotulas if not tabla_rotulas.empty else "  (no se formaron rotulas en el rango analizado)")


# %% CELDA 13: GRAFICOS DEL PUSHOVER (curva de capacidad)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(resultado_pushover['u_control'], resultado_pushover['lam'], color="C0", lw=2)
if not tabla_rotulas.empty:
    for _, fila in tabla_rotulas.iterrows():
        i_ = np.searchsorted(resultado_pushover['lam'], fila['lambda_'])
        ax.scatter(resultado_pushover['u_control'][min(i_, len(resultado_pushover['u_control'])-1)],
                   fila['lambda_'], color="C3", zorder=5)
        ax.annotate(f"R{int(fila['orden'])}",
                    (resultado_pushover['u_control'][min(i_, len(resultado_pushover['u_control'])-1)], fila['lambda_']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("Desplazamiento de control (m)")
ax.set_ylabel("Cortante basal (kN)")
ax.set_title("Curva de capacidad (pushover)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("resultado_06_curva_capacidad.png", dpi=150)


# %% CELDA 14: EXPORTAR TODOS LOS RESULTADOS A EXCEL

with pd.ExcelWriter("resultados_analisis_estructural.xlsx") as writer:
    tabla_estatico.to_excel(writer, sheet_name="Estatico_elementos")
    tabla_reacciones.to_excel(writer, sheet_name="Estatico_reacciones")
    tabla_modal.round(6).to_excel(writer, sheet_name="Modal")
    tabla_espectral.to_excel(writer, sheet_name="Espectral_envolvente")
    if not tabla_rotulas.empty:
        tabla_rotulas.to_excel(writer, sheet_name="Pushover_rotulas", index=False)
    pd.DataFrame(dict(lambda_kN=resultado_pushover['lam'],
                       u_control_m=resultado_pushover['u_control'])
                 ).to_excel(writer, sheet_name="Pushover_curva", index=False)

print("\nResultados exportados a 'resultados_analisis_estructural.xlsx' "
      "y figuras 'resultado_0X_*.png' en el directorio de trabajo actual.")

plt.show()
