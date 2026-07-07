# Análisis estructural de pórticos planos (2D) — script para Spyder

`analisis_estructural_2D.py` es un script autocontenido de Python, pensado
para abrirse y ejecutarse en **Spyder** (Anaconda), que realiza sobre un
pórtico plano de elementos viga-columna:

- **Análisis estático lineal** (cargas nodales y distribuidas).
- **Análisis modal** (frecuencias, periodos y formas modales), con
  condensación estática de los grados de libertad rotacionales sin masa.
- **Análisis modal espectral** con un espectro elástico de diseño
  parametrizable (forma NSR-10) y combinación modal **CQC** o **SRSS**.
- **Análisis no lineal** tipo *pushover* con rótulas plásticas
  concentradas (elasto-perfectamente-plásticas), método evento a evento,
  hasta la formación del mecanismo de colapso.

Todos los resultados se entregan **por elemento**, en tablas (impresas en
consola y exportadas a `resultados_analisis_estructural.xlsx`) y en
gráficos (deformada, diagramas de fuerza axial/cortante/momento, formas
modales, espectro, envolvente espectral y curva de capacidad), guardados
como archivos `.png` en el directorio de trabajo.

## Requisitos

Solo necesita una instalación estándar de **Anaconda** (incluye Spyder,
`numpy`, `scipy`, `pandas`, `matplotlib` y `openpyxl`). No se requiere
instalar nada adicional.

## Uso

1. Abra `analisis_estructural_2D.py` en Spyder.
2. Ejecute todo el script con **F5**, o celda por celda con
   **Ctrl+Enter** (las celdas están delimitadas con `# %%`).
3. El script incluye un ejemplo (pórtico de 3 niveles y 2 luces) que corre
   de inmediato. Para analizar su propia estructura, edite:
   - **Celda 2** — geometría (nodos y elementos).
   - **Celda 3** — materiales, secciones y capacidades a flexión (Mp).
   - **Celda 4** — apoyos, masas sísmicas y cargas.

   El resto del script (motor de elementos finitos, solución de los
   distintos análisis, tablas y gráficos) no requiere modificaciones.

## Alcance y limitaciones

- Elemento tipo pórtico 2D de Euler-Bernoulli (3 grados de libertad por
  nodo). No incluye deformación por cortante ni torsión.
- El análisis no lineal usa rótulas plásticas concentradas sin
  endurecimiento, sin interacción N-M-V y sin degradación cíclica: es una
  idealización simplificada, útil para prediseño y fines didácticos. Para
  diseño final utilice software especializado (ETABS, SAP2000, OpenSees)
  con curvas de capacidad reales de las secciones.
- Las cargas distribuidas se aplican en la dirección transversal local
  (uso típico: cargas de gravedad sobre vigas horizontales).
- El espectro de diseño incluido tiene la forma del espectro elástico de
  la NSR-10 (Colombia); sus parámetros (Aa, Av, Fa, Fv, I, R) son
  editables, o puede reemplazar la función `espectro_nsr10` por el
  espectro de su propia norma.

Todas las fórmulas del motor de cálculo (rigidez, transformación de ejes,
cargas equivalentes, condensación dinámica, combinación modal, rótulas
plásticas) fueron verificadas contra soluciones cerradas de la teoría de
estructuras (viga en voladizo, viga empotrada-empotrada bajo carga
uniforme, aproximación de edificio de cortante, equilibrio global de
fuerzas y momentos) antes de incluirse en el script.
