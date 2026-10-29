# zonaprop-ai

Course repository for **AI & Data Science for Economists**.

You will use the Cursor agent to download a public real-estate search into a CSV, keep one documented command, and compare the row count with the total the site announces.

## Before you start

Install these tools:

- [Node.js](https://nodejs.org/) 20 or newer (the current LTS)
- [Python](https://www.python.org/downloads/) 3.11 or newer
- The latest version of [Git](https://git-scm.com/downloads)
- The latest version of [Cursor](https://cursor.com/)

Create a [GitHub account](https://github.com/signup).

Fork [cbrutomeso/zonaprop-ai](https://github.com/cbrutomeso/zonaprop-ai), then clone your fork and open it in Cursor:

```bash
git clone https://github.com/YOUR_USERNAME/zonaprop-ai.git
cd zonaprop-ai
```

The agent instructions for this assignment are in `.cursor/` and come with the clone.

## First prompt

Paste this as the first message to the Cursor agent:

```text
Bajá a un CSV todos los resultados de esta búsqueda:
https://www.zonaprop.com.ar/casas-departamentos-ph-venta-capital-federal.html

Campos que necesito, con nombres fijos: precio en USD, barrio, ambientes, dormitorios, baños, descripción, tipo (casa, departamento, ph o terreno) y superficie total, cubierta y descubierta.

Sumá las columnas que sirvan para un análisis o un modelo. Dejá un solo comando documentado y decime cuántas filas quedaron contra el total que anuncia el sitio.
```

The CSV belongs in `data/` (or `Data/`). Those directories are gitignored, so the download stays on your machine and is not part of what you push.

## Exploratory analysis

The notebook `notebooks/eda_propiedades_caba.ipynb` explores listing prices in CABA (distributions, outliers, missing values, amenities parsed from the description, maps, and distance to subway stations).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=taller-ia --display-name="taller IA"
```

Open `notebooks/eda_propiedades_caba.ipynb` and run all cells. It reads the first CSV in `Data/` and caches barrio/subway geo files under `Data/geo/`.

## Price models

The notebook `notebooks/modelos_precio.ipynb` trains three models of `log(precio_usd)` on an 80/20 split and compares them by MAE in USD, median percentage error and R² on log price:

- Linear regression, missing values imputed with the training median
- LightGBM, missing values imputed with the training median
- LightGBM, missing values left as they are

With the `.venv` active, install the new dependencies and run all cells:

```bash
pip install -r requirements.txt
```
