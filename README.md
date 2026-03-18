# Radiator / CAC / Oil Cooler Sizing App

Streamlit app for preliminary sizing and performance checking of finned flat-tube heat exchangers, including:

- Engine jacket-water radiators
- Charge-air coolers (CAC)
- Oil coolers

The app supports elliptical / flat tubes, plate or corrugated fins, multi-pass layouts, row-by-row thermal marching, PDF report generation, and practical engineering guidance for capacity balance, pressure drop, and geometry selection.

## Main capabilities

### 1) Multi-service tube-side modes
- **Coolant liquid**
- **Charge air / CAC**
- **Oil**

### 2) Geometry and core layout
- Core width, height, depth
- Number of rows
- Inline or staggered tube arrangement
- Tube dimensions, wall thickness, pitch, corner radius
- Multi-pass tube-side layouts
  - equal tubes per pass
  - unequal pass widths

### 3) Fin options
- Plate fin
- Corrugated fin
- Louvered or non-louvered air-side treatment
- FPI and fin thickness inputs
- Fin material selection:
  - Copper
  - Brass
  - Aluminum
  - Steel

### 4) Tube material and joints
- Tube material selection:
  - Copper
  - Brass
  - Aluminum
  - Steel
- Fin-to-tube joint / bond options
  - auto by materials
  - brazed aluminum
  - lead-tin soldered
  - copper/brass brazed
  - mechanical / contact fit
  - custom override
- Bond / joint effectiveness applied to fin-side thermal contribution

### 5) Internal tube inserts
- None
- CAC internal heat-transfer fin
- Oil turbulator / strip
- Generic internal fin

User can define internal insert style and FPI. The model applies practical first-pass modifiers for:
- added internal area
- internal hydraulic diameter / free-flow area change
- internal heat-transfer enhancement
- internal pressure-drop penalty

### 6) Row-by-row marching solution
The app uses a **row-by-row air-side march** and pass-wise tube-side solution. It is intended as a practical engineering model for preliminary design and comparison studies.

- Air temperature updates row by row
- Tube-side thermal state updates through the passes
- For gas service, the app uses a **compressible pressure-drop march** with local pressure, temperature, density, and velocity updates

### 7) Thermal requirement modes
Two operating modes are provided:

- **Given required heat rejection** -> app derives the tube-side outlet target implied by the entered duty
- **Given target tube-side outlet** -> app derives the required heat rejection implied by the entered target temperature

### 8) Capacity guidance
The UI and PDF report show:
- tube-side thermal capacity rate
- air-side thermal capacity rate
- limiting side
- capacity ratio
- ideal inlet thermal limit
- duty excess or duty shortfall

This helps the user judge whether air flow, tube-side flow, or geometry should be adjusted.

### 9) PDF report generation
The app generates a downloadable PDF report with:
- executive summary
- warnings / sanity checks
- geometry, counts, and areas
- material and insert summary
- capacity guidance
- pass summary
- row-by-row appendix
- methods / assumptions

## Supported tube-side fluids

### Coolant liquid
For radiator-style service with water / glycol type use.

### Charge air / CAC
Includes compressible gas pressure-drop treatment using local pressure, density, and velocity along the pass.

### Oil
Supports practical oil mode with temperature-dependent properties and common oil grades.

#### Engine oil grades
- 0W-20
- 0W-30
- 0W-40
- 5W-30
- 5W-40
- 10W-40
- 15W-40
- 5W-50
- 10W-60

#### Hydraulic oil grades
- ISO VG 32
- ISO VG 46
- ISO VG 68

The built-in oil library is intended for engineering estimation. For final design, replace with your exact supplier property data where possible.

## Reported outputs

The app reports, among others:

- Achieved duty
- Tube-side outlet temperature
- Air outlet temperature
- Air-side pressure drop
- Tube-side pressure drop
- Tube-side velocity
- Tube-side Reynolds number
- Air-side Reynolds number
- Tube-side Nusselt number
- Pass-by-pass results
- Row-by-row thermal march results
- Tubes per row
- Total tubes
- One-fin gross area
- One-fin net area
- Total number of fins
- Total tube area
- Total net fin area
- Total air-side heat-transfer area
- Effective air-side area

## Theory level and intended use

This app is intended for:
- preliminary sizing
- concept comparison
- engineering trade-off studies
- quick selection and calibration against test data

It is **not** a substitute for:
- full CFD
- full header / manifold maldistribution modelling
- full detailed compact heat-exchanger rating software
- final certified product validation

### Important modelling notes
- The model is **1D engineering-grade**, not CFD
- Internal insert treatment is a practical approximation
- Bond / joint effect is represented as an efficiency factor, not a detailed microscopic contact model
- Oil properties are engineering estimates unless exact supplier data are entered
- Charge-air pressure-drop model is compressible but still simplified

## Installation

### Python version
Recommended: **Python 3.10+**

### Packages
Install the required packages:

```bash
pip install streamlit numpy pandas reportlab coolprop
```

If `CoolProp` is not available, parts of the app may fall back to simplified property handling.

## Running locally

```bash
streamlit run radiator_sizing_app_elliptical_tubes_compressible_gas_upgrade.py
```

## Password protection
The app is protected with a password.

Set the password either by Streamlit secrets or by environment variable.

### Option 1: Streamlit secrets
Create:

```toml
# .streamlit/secrets.toml
APP_PASSWORD = "your_password_here"
```

### Option 2: Environment variable

```bash
export APP_PASSWORD="your_password_here"
```

On Windows PowerShell:

```powershell
$env:APP_PASSWORD="your_password_here"
```

## Suggested repository structure

```text
project-root/
│
├─ radiator_sizing_app_elliptical_tubes_compressible_gas_upgrade.py
├─ README.md
├─ requirements.txt
└─ .streamlit/
   └─ secrets.toml
```

## Example requirements.txt

```txt
streamlit
numpy
pandas
reportlab
coolprop
```

## Deployment notes

This app is suitable for deployment on:
- Streamlit Community Cloud
- an internal company server
- a local engineering workstation

For Streamlit Cloud:
1. Push the Python file and `README.md` to GitHub
2. Add your password under app secrets
3. Deploy the script as the app entry point

## Recommended next improvements
Possible future upgrades:
- humid-air support on the air side
- more detailed compact-fin correlations for internal CAC fins
- dedicated oil-turbulator correlations
- header / manifold maldistribution model
- fan curve integration and operating-point matching
- calibration page using test data
- Excel export

## Disclaimer
This tool is for engineering estimation and preliminary design. Final thermal and pressure-drop performance should be validated against test data, supplier data, and detailed design review before release for production.
