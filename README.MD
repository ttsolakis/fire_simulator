## 🔥 Fire Simulator

A Python package for simulating wildfire spread on raster grids (GeoTIFF), with simple visualization using Matplotlib.  
The simulator uses **Copernicus Global Land Cover (LCM-10)** maps to define vegetation and flammability.

---

## ⚙️ Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/ttsolakis/fire_simulator.git
cd fire_simulator

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install system dependencies for [`rasterio`](https://rasterio.readthedocs.io/):

```bash
# Linux
sudo apt update
sudo apt install -y gdal-bin libgdal-dev
```

Then install Python dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🌍 Downloading Land Cover Data (Copernicus Browser)

To simulate realistic fire spread, you’ll need a **land cover map**.  
We recommend the **Copernicus Global Dynamic Land Cover (2020, 10 m)** dataset.

1. Open Copernicus Browser here:  
   👉 [Copernicus Global Land Cover 2020](https://land.copernicus.eu/en/products/global-dynamic-land-cover/land-cover-2020-raster-10-m-global-annual#download)
2. Click **“Copernicus Browser”**  
3. Login (create a free account if needed)  
4. From **Layers**, choose **LCM-10**  
5. On the right toolbar, click **“Download Image”**  
6. In the wizard, select:  
   - **Type** → `Analytical`  
   - **Show Logo** → `No`  
   - **Image Format** → `TIFF (32-bit float)`  
   - **Image Resolution** → `High` (better quality)  
   - **Coordinate System** → `WGS 84 (EPSG:4326)`  
   - **Clip extra bands** → `No`  
   - **Layers** → ✅ **Raw LCM-10 only**  
7. Download the `.tif` file and place it in `fire_simulator/data/`

---

## ▶️ Usage

Run the simulator from the command line **from one directory above the package**  
(e.g., if your path is `~/dev_ws/src/fire_simulator`, then run from `~/dev_ws/src`):

```bash
python3 -m fire_simulator
```

Once the map window opens, click any pixel to ignite that point. The fire will then spread according to the model.

---

## 📂 Project Structure

```
fire_simulator/
├── data/                  # Example TIFF maps
├── fire_grid.py           # Grid representation
├── fire_model.py          # Fire spread rules
├── fire_simulation.py     # Simulation runner
├── visualize.py           # Visualization utilities
├── __main__.py            # CLI entry point
├── requirements.txt       # Dependencies
└── setup.py               # Package definition
```
---

## 📝 License

MIT License © 2025 **Anastasios (Tasos) Tsolakis**
