# app.py — Elliptical Radiator Sizer (Zukauskas + Colburn-j), coolant parallel across rows
import math, io, textwrap
import hmac
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Radiator Sizer", layout="wide")

# ============================================================================
# PASSWORD PROTECTION
# ============================================================================
def check_password():
    """Return True only when the correct app password is entered."""
    expected_password = None
    try:
        if "APP_PASSWORD" in st.secrets:
            expected_password = st.secrets["APP_PASSWORD"]
        elif "app_password" in st.secrets:
            expected_password = st.secrets["app_password"]
    except Exception:
        expected_password = None

    if expected_password is None:
        import os
        expected_password = os.environ.get("APP_PASSWORD")

    if not expected_password:
        st.error("App password is not configured. Add APP_PASSWORD to Streamlit Secrets.")
        st.stop()

    def password_entered():
        entered = st.session_state.get("password", "")
        if hmac.compare_digest(str(entered), str(expected_password)):
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- Calibration multipliers: session-state defaults & apply hooks ---
if 'k_UA_default' not in st.session_state: st.session_state['k_UA_default'] = 1.0
if 'k_DP_default' not in st.session_state: st.session_state['k_DP_default'] = 1.0
# When Apply buttons are pressed in the fitter, set *_new and *_apply flags
if st.session_state.get('apply_k_UA', False):
    # Clear the existing widget value and update default
    st.session_state.pop('k_UA', None)
    st.session_state['k_UA_default'] = float(st.session_state.get('k_UA_new', 1.0))
    st.session_state['apply_k_UA'] = False
if st.session_state.get('apply_k_DP', False):
    st.session_state.pop('k_DP', None)
    st.session_state['k_DP_default'] = float(st.session_state.get('k_DP_new', 1.0))
    st.session_state['apply_k_DP'] = False


# ---- Optional thermophys libs ----
try:
    from CoolProp.CoolProp import PropsSI
    from CoolProp.HumidAirProp import HAPropsSI
except Exception:
    PropsSI = None
    HAPropsSI = None

# ---- PDF ----
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# -------------------- Helpers --------------------
def ellipse_perimeter(a: float, b: float) -> float:
    h = ((a-b)**2)/((a+b)**2)
    return math.pi*(a+b)*(1 + (3*h)/(10+math.sqrt(4-3*h)))

def ellipse_area(a: float, b: float) -> float:
    return math.pi*a*b

def rounded_rect_area(width: float, height: float, radius: float) -> float:
    radius = max(0.0, min(radius, 0.5*min(width, height)))
    return width*height - (4.0 - math.pi)*(radius**2)

def rounded_rect_perimeter(width: float, height: float, radius: float) -> float:
    radius = max(0.0, min(radius, 0.5*min(width, height)))
    return 2.0*(width + height) + 2.0*(math.pi - 4.0)*radius

def air_properties(T_C: float, RH_frac: float, P_Pa: float = 101325.0) -> Dict[str, float]:
    T_K = T_C + 273.15
    if HAPropsSI is not None:
        try:
            rho = HAPropsSI('Rho','T',T_K,'P',P_Pa,'RH',RH_frac)
            cp  = HAPropsSI('C','T',T_K,'P',P_Pa,'RH',RH_frac)
            k   = HAPropsSI('K','T',T_K,'P',P_Pa,'RH',RH_frac)
            mu  = HAPropsSI('MU','T',T_K,'P',P_Pa,'RH',RH_frac)
        except Exception:
            rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.95e-5
    else:
        rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.95e-5
    Pr = cp*mu/max(k,1e-12)
    return {"rho":rho,"cp":cp,"k":k,"mu":mu,"Pr":Pr}


def sat_pressure_water_Pa(T_C: float) -> float:
    # Antoine correlation for water, valid enough for engineering warning use around 1–150 C
    T_C = float(max(1.0, min(150.0, T_C)))
    A, B, C = 8.14019, 1810.94, 244.485  # P in mmHg
    P_mmHg = 10**(A - B/(C + T_C))
    return P_mmHg * 133.322368

def liquid_water_props(Tc: float) -> Dict[str, float]:
    """
    Stable liquid-water approximation for radiator calculations.
    Uses simple interpolants / empirical forms and intentionally avoids
    gas-phase flips near 100 C that can happen when calling water at 1 atm.
    """
    T = float(max(0.0, min(150.0, Tc)))
    T_pts = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 150.0], dtype=float)
    rho_pts = np.array([999.84, 998.20, 992.20, 983.20, 971.80, 958.40, 943.10, 926.10, 917.00], dtype=float)
    cp_pts  = np.array([4217.0, 4182.0, 4179.0, 4185.0, 4196.0, 4216.0, 4245.0, 4280.0, 4300.0], dtype=float)
    k_pts   = np.array([0.561, 0.598, 0.628, 0.651, 0.668, 0.679, 0.684, 0.684, 0.682], dtype=float)

    rho = float(np.interp(T, T_pts, rho_pts))
    cp = float(np.interp(T, T_pts, cp_pts))
    k = float(np.interp(T, T_pts, k_pts))

    T_K = T + 273.15
    # Andrade-type viscosity for liquid water, Pa·s
    mu = 2.414e-5 * (10 ** (247.8 / max(T_K - 140.0, 1e-6)))
    Pr = cp * mu / max(k, 1e-12)
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "Pr": Pr}

MATERIAL_K_W_MK = {
    "Copper": 385.0,
    "Brass": 120.0,
    "Aluminum": 205.0,
    "Steel": 45.0,
}

def material_k(material_name: str) -> float:
    return MATERIAL_K_W_MK.get(str(material_name), 205.0)

JOINT_TYPE_OPTIONS = [
    "Auto from materials",
    "Brazed aluminum",
    "Lead-tin soldered",
    "Copper/brass brazed",
    "Mechanical / contact fit",
    "Custom / override",
]

def resolve_joint_type_and_effectiveness(tube_material: str, fin_material: str, joint_type_choice: str) -> tuple[str, float]:
    """Return a resolved joint type label and a practical bond-effectiveness factor.
    This is an engineering calibration factor, not an explicit solder/braze slab resistance.
    """
    t = str(tube_material)
    f = str(fin_material)
    auto_choice = str(joint_type_choice)
    if auto_choice == "Auto from materials":
        if t == "Aluminum" and f == "Aluminum":
            return "Brazed aluminum", 0.99
        if {t, f}.issubset({"Copper", "Brass"}):
            return "Lead-tin soldered", 0.95
        return "Mechanical / contact fit", 0.92
    defaults = {
        "Brazed aluminum": 0.99,
        "Lead-tin soldered": 0.95,
        "Copper/brass brazed": 0.97,
        "Mechanical / contact fit": 0.90,
        "Custom / override": 0.95,
    }
    return auto_choice, defaults.get(auto_choice, 0.95)

def build_internal_insert_model(has_internal_fins: bool,
                                insert_type: str,
                                id_w: float,
                                id_d: float,
                                tube_length_m: float,
                                fin_count_per_tube: int,
                                fin_thickness_m: float,
                                Ai_plain: float,
                                Pi_plain: float,
                                fin_fpi: float) -> Dict[str, float | str]:
    """Approximate internal-insert effects.
    - CAC heat-transfer fin: more real added area + moderate h boost + moderate dp penalty.
    - Oil turbulator: less true added area credited, stronger mixing boost + stronger dp penalty.
    - Generic: in between.
    """
    base = {
        'insert_type_resolved': 'None',
        'blockage_frac': 0.0,
        'Ai_flow_one': Ai_plain,
        'Pi_heat_one': Pi_plain,
        'A_internal_added_one_tube': 0.0,
        'internal_area_ratio_geom': 1.0,
        'internal_area_ratio_equiv': 1.0,
        'internal_h_enhancement': 1.0,
        'internal_dp_multiplier': 1.0,
        'wetted_area_credit_factor': 1.0,
    }
    if not has_internal_fins:
        return base

    count = max(int(fin_count_per_tube), 1)
    waviness = 1.10
    raw_added_area = 2.0 * count * id_d * tube_length_m * waviness
    raw_added_perimeter = 2.0 * count * id_d * waviness
    plain_area = max(Pi_plain * tube_length_m, 1e-12)

    insert_type = str(insert_type)
    if insert_type == "CAC internal heat-transfer fin":
        blockage_frac = min(0.72, count * fin_thickness_m / max(id_w, 1e-12))
        area_credit = 1.00
        h_mult = min(2.8, 1.0 + 0.06 * fin_fpi)
        dp_mult = min(6.5, 1.0 + 0.12 * fin_fpi)
    elif insert_type == "Oil turbulator / strip":
        blockage_frac = min(0.62, 0.75 * count * fin_thickness_m / max(id_w, 1e-12))
        area_credit = 0.30
        h_mult = min(4.5, 1.0 + 0.11 * fin_fpi)
        dp_mult = min(12.0, 1.0 + 0.24 * fin_fpi)
    else:
        blockage_frac = min(0.70, 0.90 * count * fin_thickness_m / max(id_w, 1e-12))
        area_credit = 0.65
        h_mult = min(3.3, 1.0 + 0.08 * fin_fpi)
        dp_mult = min(8.5, 1.0 + 0.16 * fin_fpi)

    Ai_flow_one = Ai_plain * max(0.15, 1.0 - blockage_frac)
    Pi_heat_one = Pi_plain + area_credit * raw_added_perimeter
    A_internal_added_one_tube = area_credit * raw_added_area
    A_i_heat_one = Pi_heat_one * tube_length_m
    area_ratio_geom = A_i_heat_one / plain_area
    area_ratio_equiv = max(1.0, area_ratio_geom)

    return {
        'insert_type_resolved': insert_type,
        'blockage_frac': blockage_frac,
        'Ai_flow_one': Ai_flow_one,
        'Pi_heat_one': Pi_heat_one,
        'A_internal_added_one_tube': A_internal_added_one_tube,
        'internal_area_ratio_geom': area_ratio_geom,
        'internal_area_ratio_equiv': area_ratio_equiv,
        'internal_h_enhancement': h_mult,
        'internal_dp_multiplier': dp_mult,
        'wetted_area_credit_factor': area_credit,
    }

def dry_air_props(T_C: float, P_abs_Pa: float = 101325.0) -> Dict[str, float]:
    T_K = T_C + 273.15
    R = 287.05
    rho = max(P_abs_Pa, 1000.0) / max(R*T_K, 1e-12)
    cp = 1007.0
    mu = 1.716e-5 * ((T_K/273.15)**1.5) * (273.15 + 111.0) / (T_K + 111.0)
    k = 0.0241 * ((T_K/273.15)**0.9)
    Pr = cp * mu / max(k, 1e-12)
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "Pr": Pr}

ENGINE_OIL_LIBRARY = {
    # Representative values for common grades. Exact numbers vary by brand/product.
    "0W-20": {"rho15": 843.0, "nu40": 47.0,   "nu100": 8.9,   "cp40": 2100.0, "k40": 0.145},
    "0W-30": {"rho15": 845.0, "nu40": 58.0,   "nu100": 10.8,  "cp40": 2100.0, "k40": 0.145},
    "0W-40": {"rho15": 845.0, "nu40": 75.0,   "nu100": 13.3,  "cp40": 2080.0, "k40": 0.145},
    "5W-30": {"rho15": 851.0, "nu40": 73.0,   "nu100": 12.3,  "cp40": 2100.0, "k40": 0.145},
    "5W-40": {"rho15": 851.5, "nu40": 80.53,  "nu100": 13.44, "cp40": 2080.0, "k40": 0.145},
    "10W-40":{"rho15": 866.0, "nu40": 100.0,  "nu100": 14.4,  "cp40": 2050.0, "k40": 0.145},
    "15W-40":{"rho15": 870.0, "nu40": 115.0,  "nu100": 15.4,  "cp40": 2050.0, "k40": 0.145},
    "5W-50": {"rho15": 855.0, "nu40": 105.0,  "nu100": 17.5,  "cp40": 2050.0, "k40": 0.145},
    "10W-60":{"rho15": 860.0, "nu40": 160.0,  "nu100": 24.0,  "cp40": 2000.0, "k40": 0.145},
}

HYDRAULIC_OIL_LIBRARY = {
    "ISO VG 32": {"rho15": 843.0, "nu40": 32.0, "nu100": 6.0,   "cp40": 1950.0, "k40": 0.130},
    "ISO VG 46": {"rho15": 846.0, "nu40": 46.0, "nu100": 8.2,   "cp40": 1950.0, "k40": 0.130},
    "ISO VG 68": {"rho15": 851.0, "nu40": 68.0, "nu100": 10.37, "cp40": 1950.0, "k40": 0.130},
}

def walther_nu_cSt(T_C: float, nu40_cSt: float, nu100_cSt: float) -> float:
    """ASTM D341 / Walther-style interpolation between kinematic viscosities at 40 C and 100 C."""
    T_K = max(T_C + 273.15, 220.0)
    nu40 = max(float(nu40_cSt), 0.6)
    nu100 = max(float(nu100_cSt), 0.6)
    X1 = math.log10(313.15)
    X2 = math.log10(373.15)
    Y1 = math.log10(math.log10(nu40 + 0.7))
    Y2 = math.log10(math.log10(nu100 + 0.7))
    B = (Y1 - Y2) / max(X2 - X1, 1e-12)
    A = Y1 + B * X1
    Y = A - B * math.log10(T_K)
    return max((10 ** (10 ** Y)) - 0.7, 0.5)

def oil_props_constant(T_C: float, rho_ref: float = 850.0, cp_ref: float = 2100.0,
                       k_ref: float = 0.13, mu_mPas: float = 25.0) -> Dict[str, float]:
    rho = max(rho_ref * (1.0 - 6.5e-4*(T_C - 40.0)), 0.6*rho_ref)
    cp = max(cp_ref * (1.0 + 4.0e-4*(T_C - 40.0)), 500.0)
    k = max(k_ref * (1.0 - 8.0e-4*(T_C - 40.0)), 0.05)
    mu = max(mu_mPas, 0.2) * 1e-3
    Pr = cp * mu / max(k, 1e-12)
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "Pr": Pr}

def oil_props_grade(T_C: float, grade_data: Dict[str, float], oil_family: str = 'Engine oil') -> Dict[str, float]:
    rho15 = float(grade_data.get('rho15', 850.0))
    nu40 = float(grade_data.get('nu40', 46.0))
    nu100 = float(grade_data.get('nu100', 8.0))
    cp40 = float(grade_data.get('cp40', 2000.0))
    k40 = float(grade_data.get('k40', 0.13))
    beta = 7.0e-4 if 'Hydraulic' in str(oil_family) else 6.5e-4
    rho = max(rho15 * (1.0 - beta * (T_C - 15.0)), 0.55 * rho15)
    nu_cSt = walther_nu_cSt(T_C, nu40, nu100)
    mu = nu_cSt * 1e-6 * rho
    cp = max(cp40 * (1.0 + 1.2e-3 * (T_C - 40.0)), 800.0)
    k = max(k40 * (1.0 - 6.0e-4 * (T_C - 40.0)), 0.07)
    Pr = cp * mu / max(k, 1e-12)
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "Pr": Pr, "nu_cSt": nu_cSt}

def get_tube_side_props(service_mode: str, T_C: float, P_abs_Pa: float,
                        coolant_name: str = 'Water', glycol_pct: float = 0.0,
                        oil_inputs: Dict[str, float] | None = None) -> Dict[str, float]:
    if service_mode == 'Coolant liquid':
        return get_coolant_props(coolant_name, T_C, glycol_pct, P_abs_Pa)
    if service_mode == 'Charge air / CAC':
        return dry_air_props(T_C, P_abs_Pa)
    oil_inputs = oil_inputs or {}
    if oil_inputs.get('model') == 'grade_library':
        return oil_props_grade(T_C, oil_inputs, oil_family=str(oil_inputs.get('family', 'Engine oil')))
    return oil_props_constant(
        T_C,
        rho_ref=float(oil_inputs.get('rho', 850.0)),
        cp_ref=float(oil_inputs.get('cp', 2100.0)),
        k_ref=float(oil_inputs.get('k', 0.13)),
        mu_mPas=float(oil_inputs.get('mu_mPas', 25.0)),
    )

def coolant_properties(fluid: str, T_C: float, P_abs_Pa: float = 250000.0) -> Dict[str, float]:
    """
    Coolant properties used by the radiator model.
    For water we force a liquid-property path so hot pressurised coolant does not
    accidentally get evaluated as vapour because of a 1 atm assumption.
    """
    fluid = str(fluid).strip()
    if fluid.lower() == 'water':
        return liquid_water_props(T_C)

    T_K = T_C + 273.15
    if PropsSI is not None:
        try:
            rho = PropsSI('D', 'T', T_K, 'P', max(P_abs_Pa, 101325.0), fluid)
            cp = PropsSI('C', 'T', T_K, 'P', max(P_abs_Pa, 101325.0), fluid)
            k = PropsSI('L', 'T', T_K, 'P', max(P_abs_Pa, 101325.0), fluid)
            mu = PropsSI('V', 'T', T_K, 'P', max(P_abs_Pa, 101325.0), fluid)
        except Exception:
            base = liquid_water_props(T_C)
            rho, cp, k, mu = base['rho'], base['cp'], base['k'], base['mu']
    else:
        base = liquid_water_props(T_C)
        rho, cp, k, mu = base['rho'], base['cp'], base['k'], base['mu']

    Pr = cp * mu / max(k, 1e-12)
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "Pr": Pr}

def coolant_props(Tc, glycol_type, pct, P_abs_Pa: float = 250000.0):
    # backward-compatible wrapper if rest of app uses older naming elsewhere
    return coolant_properties('Water', Tc, P_abs_Pa)

def water_props(Tc, P_abs_Pa: float = 250000.0):
    return coolant_properties('Water', Tc, P_abs_Pa)

def simple_meg_props(Tc, pct, P_abs_Pa: float = 250000.0):
    base = liquid_water_props(Tc)
    x = max(0.0, min(0.7, pct/100.0))
    cp = base['cp']*(1-0.18*x)
    mu = base['mu']*(1+6.0*x+20.0*x*x)
    k = base['k']*(1-0.12*x)
    rho = base['rho']*(1+0.08*x)
    return {
        'rho': rho,
        'cp': cp,
        'k':  k,
        'mu': mu,
        'Pr': cp*mu/max(k,1e-12)
    }

def simple_mpg_props(Tc, pct, P_abs_Pa: float = 250000.0):
    base = liquid_water_props(Tc)
    x = max(0.0, min(0.7, pct/100.0))
    cp = base['cp']*(1-0.16*x)
    mu = base['mu']*(1+7.0*x+22.0*x*x)
    k = base['k']*(1-0.10*x)
    rho = base['rho']*(1+0.07*x)
    return {
        'rho': rho,
        'cp': cp,
        'k':  k,
        'mu': mu,
        'Pr': cp*mu/max(k,1e-12)
    }

def get_coolant_props(coolant_name: str, T_C: float, glycol_pct: float = 0.0, P_abs_Pa: float = 250000.0):
    if coolant_name == 'Water':
        return water_props(T_C, P_abs_Pa)
    elif coolant_name == 'MEG/Water':
        return simple_meg_props(T_C, glycol_pct, P_abs_Pa)
    elif coolant_name == 'MPG/Water':
        return simple_mpg_props(T_C, glycol_pct, P_abs_Pa)
    return water_props(T_C, P_abs_Pa)

    if coolant_name == 'Water':
        return water_props(T_C)
    elif coolant_name == 'MEG/Water':
        return simple_meg_props(T_C, glycol_pct)
    elif coolant_name == 'MPG/Water':
        return simple_mpg_props(T_C, glycol_pct)
    return water_props(T_C)

def reynolds(rho, v, Dh, mu):
    return rho*v*Dh/max(mu,1e-12)

def gnielinski(Re, Pr):
    if Re < 2300:
        return 3.66
    f = (0.79*math.log(max(Re, 1e-9)) - 1.64)**(-2)
    return (f/8)*(Re-1000)*Pr / (1 + 12.7*math.sqrt(f/8)*(Pr**(2/3)-1))

def zukauskas(Re, Pr, n_rows=1):
    if Re < 1e2:
        C, m = 0.9, 0.4
    elif Re < 1e3:
        C, m = 0.52, 0.5
    elif Re < 2e5:
        C, m = 0.27, 0.63
    else:
        C, m = 0.033, 0.8
    n = 0.36 if Pr <= 10 else 0.25
    C2 = 1.0
    if n_rows == 1: C2 = 0.64
    elif n_rows == 2: C2 = 0.76
    elif n_rows == 3: C2 = 0.84
    elif n_rows == 4: C2 = 0.89
    elif n_rows == 5: C2 = 0.92
    elif n_rows == 6: C2 = 0.95
    elif n_rows >= 7: C2 = 1.0
    Nu = C*(Re**m)*(Pr**n)*C2
    return Nu, C2

def tube_bank_velocity(face_velocity, free_area_ratio):
    return face_velocity / max(free_area_ratio, 1e-9)

def friction_blasius(Re):
    if Re < 2300:
        return 64.0/max(Re,1e-12)
    return 0.3164/(max(Re,1e-12)**0.25)

def nu_internal_tube(Re, Pr, Dh, L):
    """Internal-flow Nusselt number with a simple transition blend.
    Laminar branch uses a developing-flow correction around the 3.66 fully developed limit.
    Turbulent branch uses Gnielinski. Between Re=2300 and 4000, the two are blended linearly.
    """
    Re_eff = max(Re, 1e-12)
    Pr_eff = max(Pr, 1e-12)
    x = max((Dh/max(L, 1e-12))*Re_eff*Pr_eff, 1e-12)
    nu_lam = 3.66 + (0.0668*x)/(1.0 + 0.04*(x**(2.0/3.0)))
    nu_turb = max(gnielinski(max(Re_eff, 4000.0), Pr_eff), 3.66)
    if Re_eff <= 2300.0:
        return nu_lam, 'laminar/developing'
    if Re_eff >= 4000.0:
        return nu_turb, 'turbulent (Gnielinski)'
    w = (Re_eff - 2300.0)/(4000.0 - 2300.0)
    return (1.0 - w)*nu_lam + w*nu_turb, 'transitional blend'

def f_internal_tube(Re):
    """Darcy friction factor with a transition blend between laminar and Blasius."""
    Re_eff = max(Re, 1e-12)
    f_lam = 64.0/Re_eff
    f_turb = 0.3164/(max(Re_eff, 4000.0)**0.25)
    if Re_eff <= 2300.0:
        return f_lam, 'laminar'
    if Re_eff >= 4000.0:
        return f_turb, 'turbulent'
    w = (Re_eff - 2300.0)/(4000.0 - 2300.0)
    return (1.0 - w)*f_lam + w*f_turb, 'transitional blend'

def colburn_j_corrugated(Re, fin_pitch, Dh_air, pitch_l, pitch_w, louver_pitch=0.0, louver_angle_deg=0.0,
                         Cj=0.12, mj=0.30, aj=0.05, bj=0.03, cj=0.08):
    theta = math.radians(max(louver_angle_deg, 0.0))
    pf_over_dh = max(fin_pitch/max(Dh_air,1e-12), 1e-9)
    pl_over_pf = max(pitch_l/max(fin_pitch,1e-12), 1e-9)
    base = Cj * (max(Re,1e-12)**(-mj)) * (pf_over_dh**aj) * (pl_over_pf**bj)
    if louver_angle_deg > 0 and louver_pitch > 0:
        lppf = max(louver_pitch/max(fin_pitch,1e-12), 1e-9)
        base *= (max(math.sin(theta),1e-6)**cj) * (lppf**(-0.02))
    return base

def fin_friction_corr(Re, Cf=1.5, mf=0.35):
    return Cf * (max(Re,1e-12)**(-mf))

def fin_efficiency_rect(h, kf, t, Lc):
    m = math.sqrt(max(2*h/(max(kf,1e-12)*max(t,1e-12)), 1e-12))
    x = m*max(Lc,1e-12)
    if x < 1e-8:
        return 1.0
    return math.tanh(x)/x

def crossflow_mixed_unmixed_effectiveness(NTU, Cr):
    Cr = max(min(Cr, 0.999999), 1e-12)
    return 1.0 - math.exp((math.exp(-Cr*NTU)-1.0)/Cr)

def count_tubes_per_row(core_width_mm, pitch_w_mm, od_w_mm):
    if core_width_mm < od_w_mm:
        return 1
    return max(1, int(math.floor((core_width_mm - od_w_mm)/pitch_w_mm) + 1))

def split_equal_tubes(total_tubes_per_row, n_passes):
    base = total_tubes_per_row // n_passes
    rem = total_tubes_per_row % n_passes
    arr = [base]*n_passes
    for i in range(rem):
        arr[i] += 1
    return arr

def widths_from_tubes(tubes_per_row_pass, total_width_mm):
    s = max(sum(tubes_per_row_pass), 1)
    return [total_width_mm*(n/s) for n in tubes_per_row_pass]

def tubes_from_widths(pass_widths_mm, total_tubes_per_row, total_width_mm):
    raw = [w/total_width_mm*total_tubes_per_row for w in pass_widths_mm]
    ints = [max(1, int(math.floor(x))) for x in raw]
    while sum(ints) < total_tubes_per_row:
        frac = [raw[i]-ints[i] for i in range(len(raw))]
        k = max(range(len(raw)), key=lambda i: frac[i])
        ints[k] += 1
    while sum(ints) > total_tubes_per_row:
        k = max(range(len(ints)), key=lambda i: ints[i])
        if ints[k] > 1:
            ints[k] -= 1
        else:
            break
    return ints

# -------------------- UI --------------------
st.title("🚗 Radiator Sizing — NTU/ε & ΔP (Zukauskas + Colburn-j + Kays-London)")
st.caption("Elliptical / obround tubes, plate or corrugated fins, row-by-row model, multipass coolant routing, and PDF reporting")

with st.sidebar:
    st.header("Inputs")

    st.subheader("Geometry")
    core_width_mm  = st.number_input("Core width (mm)", min_value=100.0, value=1250.0, step=10.0)
    core_height_mm = st.number_input("Core height (mm)", min_value=100.0, value=1600.0, step=10.0)
    core_depth_mm  = st.number_input("Core depth (mm)", min_value=10.0, value=124.0, step=1.0)
    n_rows         = st.number_input("Number of rows", min_value=1, value=8, step=1)
    arrangement    = st.selectbox("Tube arrangement", ["Inline","Staggered"], index=0)

    st.subheader("Tube geometry")
    od_d_mm        = st.number_input("Tube outside depth / airflow dimension (mm)", min_value=1.0, value=12.22, step=0.01)
    od_w_mm        = st.number_input("Tube outside width / minor dimension (mm)", min_value=0.5, value=2.57, step=0.01)
    tube_thk_mm    = st.number_input("Tube wall thickness (mm)", min_value=0.02, value=0.12, step=0.01)
    tube_corner_radius_mm = st.number_input("Tube outside corner radius (mm)", min_value=0.0,
                                             max_value=float(0.5*od_w_mm), value=min(1.27, float(0.5*od_w_mm)), step=0.01)
    pitch_w_mm     = st.number_input("Tube transverse pitch across width (mm)", min_value=1.0, value=13.5, step=0.1)
    pitch_l_mm     = st.number_input("Tube longitudinal pitch by row / depth (mm)", min_value=1.0, value=15.5, step=0.1)

    st.subheader("Fin geometry")
    fin_style      = st.selectbox("Fin style", ["Plate fin","Corrugated fin"], index=0)
    fin_louvered   = st.selectbox("Louvering", ["Non-louvered","Louvered"], index=0)
    fpi            = st.number_input("Fins per inch (FPI)", min_value=1.0, value=9.0, step=0.1)
    fin_thk_mm     = st.number_input("Fin thickness (mm)", min_value=0.02, value=0.12, step=0.01)
    fin_material   = st.selectbox("Fin material", ["Copper","Brass","Aluminum","Steel"], index=0)
    if fin_louvered == "Louvered":
        louver_pitch_mm = st.number_input("Louver pitch (mm)", min_value=0.1, value=1.4, step=0.1)
        louver_angle_deg = st.number_input("Louver angle (deg)", min_value=1.0, value=28.0, step=0.5)
    else:
        louver_pitch_mm = 0.0
        louver_angle_deg = 0.0

    st.subheader("Tube material, joints & internal inserts")
    tube_material  = st.selectbox("Tube material", ["Copper","Brass","Aluminum","Steel"], index=2)
    joint_type_choice = st.selectbox("Fin-to-tube joint type", JOINT_TYPE_OPTIONS, index=0,
                                     help="Uses a practical bond/joint effectiveness factor rather than an explicit thin solder/braze layer resistance.")
    resolved_joint_type_default, joint_effectiveness_default = resolve_joint_type_and_effectiveness(tube_material, fin_material, joint_type_choice)
    joint_effectiveness = st.number_input("Fin-to-tube bond effectiveness (-)", min_value=0.50, max_value=1.00,
                                          value=float(joint_effectiveness_default), step=0.01,
                                          help="Applied to the fin contribution only. Good brazed aluminum joints are typically close to 1.0; soldered copper/brass joints are usually a little lower.")
    st.caption(f"Resolved joint assumption: {resolved_joint_type_default} (default effectiveness ≈ {joint_effectiveness_default:.2f}).")
    internal_fins_in_tube = st.radio("Internal insert inside tube?", ["No", "Yes"], index=0, horizontal=True)
    if internal_fins_in_tube == "Yes":
        internal_insert_type = st.selectbox("Internal insert type", ["CAC internal heat-transfer fin", "Oil turbulator / strip", "Generic internal fin"], index=0)
        internal_fin_style = "Zigzag fin / insert"
        internal_fin_fpi = st.number_input("Internal insert density (FPI)", min_value=1.0, value=12.0, step=0.5)
        internal_fin_thk_mm = st.number_input("Internal insert thickness (mm)", min_value=0.02, value=max(0.02, float(tube_thk_mm)), step=0.01)
        st.caption("CAC mode credits more real added area. Oil-turbulator mode credits less real area but more mixing/HTC boost and a stronger ΔP penalty.")
    else:
        internal_insert_type = "None"
        internal_fin_style = "None"
        internal_fin_fpi = 0.0
        internal_fin_thk_mm = tube_thk_mm

    st.subheader("Tube-side service")
    tube_side_service = st.selectbox("Tube-side fluid", ["Coolant liquid","Charge air / CAC","Oil"], index=0)
    coolant_name = "Water"
    glycol_pct = 0.0
    tube_flow_basis = "volumetric"
    charge_air_mdot_kg_s = None
    oil_family = "Engine oil"
    oil_grade = "10W-40"
    oil_use_grade_library = True
    oil_manual_override = False
    oil_nu40_cSt = 100.0
    oil_nu100_cSt = 14.4
    oil_density_kg_m3 = 850.0
    oil_cp_J_kgK = 2100.0
    oil_k_W_mK = 0.13
    oil_mu_mPas = 25.0
    if tube_side_service == "Coolant liquid":
        coolant_name   = st.selectbox("Coolant", ["Water","MEG/Water","MPG/Water"], index=0)
        glycol_pct     = st.number_input("Glycol concentration (%)", min_value=0.0, value=0.0, step=1.0)
        T_cool_in_C    = st.number_input("Coolant inlet temperature (°C)", value=95.0, step=0.1)
        coolant_Vdot_Lps = st.number_input("Coolant flow rate (L/s)", min_value=0.01, value=7.5, step=0.01)
        coolant_pressure_g_kPa = st.number_input("Coolant operating pressure (kPa, gauge)", min_value=0.0, value=100.0, step=5.0,
                                                 help="Used for a liquid-phase sanity check. Properties are evaluated with a liquid coolant model to avoid vapour-density glitches at high temperature.")
        tube_side_label = "Coolant"
    elif tube_side_service == "Charge air / CAC":
        T_cool_in_C    = st.number_input("Charge-air inlet temperature (°C)", value=180.0, step=0.1)
        charge_air_mdot_kg_s = st.number_input("Charge-air mass flow (kg/s)", min_value=0.01, value=3.0, step=0.01)
        coolant_pressure_g_kPa = st.number_input("Charge-air inlet pressure (kPa, gauge)", min_value=0.0, value=150.0, step=5.0)
        coolant_Vdot_Lps = 0.0
        tube_flow_basis = "mass"
        tube_side_label = "Charge air"
    else:
        T_cool_in_C    = st.number_input("Oil inlet temperature (°C)", value=110.0, step=0.1)
        coolant_Vdot_Lps = st.number_input("Oil flow rate (L/s)", min_value=0.001, value=2.0, step=0.01)
        coolant_pressure_g_kPa = st.number_input("Oil inlet pressure (kPa, gauge)", min_value=0.0, value=300.0, step=5.0)
        oil_family = st.selectbox("Oil family", ["Engine oil", "Hydraulic oil"], index=0)
        if oil_family == "Engine oil":
            oil_grade = st.selectbox("Engine oil grade", list(ENGINE_OIL_LIBRARY.keys()), index=5)
            oil_grade_defaults = ENGINE_OIL_LIBRARY[oil_grade]
        else:
            oil_grade = st.selectbox("Hydraulic oil grade", list(HYDRAULIC_OIL_LIBRARY.keys()), index=1)
            oil_grade_defaults = HYDRAULIC_OIL_LIBRARY[oil_grade]
        oil_use_grade_library = st.checkbox("Use temperature-dependent oil grade library", value=True,
                                            help="Uses representative rho / nu@40C / nu@100C / cp / k values for the selected oil grade and updates properties with temperature as the model marches.")
        oil_manual_override = st.checkbox("Override oil grade defaults manually", value=False)
        if oil_manual_override or not oil_use_grade_library:
            oil_density_kg_m3 = st.number_input("Oil density at 15°C (kg/m³)", min_value=500.0, value=float(oil_grade_defaults.get('rho15', 850.0)), step=5.0)
            oil_nu40_cSt = st.number_input("Oil kinematic viscosity at 40°C (cSt)", min_value=1.0, value=float(oil_grade_defaults.get('nu40', 46.0)), step=1.0)
            oil_nu100_cSt = st.number_input("Oil kinematic viscosity at 100°C (cSt)", min_value=1.0, value=float(oil_grade_defaults.get('nu100', 8.0)), step=0.1)
            oil_cp_J_kgK = st.number_input("Oil cp at 40°C (J/kg·K)", min_value=500.0, value=float(oil_grade_defaults.get('cp40', 2100.0)), step=50.0)
            oil_k_W_mK = st.number_input("Oil conductivity at 40°C (W/m·K)", min_value=0.05, value=float(oil_grade_defaults.get('k40', 0.13)), step=0.005)
            oil_mu_mPas = st.number_input("Oil viscosity manual fallback (mPa·s)", min_value=0.5, value=25.0, step=0.5)
        else:
            oil_density_kg_m3 = float(oil_grade_defaults.get('rho15', 850.0))
            oil_nu40_cSt = float(oil_grade_defaults.get('nu40', 46.0))
            oil_nu100_cSt = float(oil_grade_defaults.get('nu100', 8.0))
            oil_cp_J_kgK = float(oil_grade_defaults.get('cp40', 2100.0))
            oil_k_W_mK = float(oil_grade_defaults.get('k40', 0.13))
            oil_mu_mPas = max(oil_nu40_cSt * oil_density_kg_m3 * 1e-3, 0.5)
            st.caption(f"Using representative {oil_family.lower()} grade data for {oil_grade}: rho15={oil_density_kg_m3:.1f} kg/m³, nu40={oil_nu40_cSt:.1f} cSt, nu100={oil_nu100_cSt:.2f} cSt.")
        tube_side_label = "Oil"
    coolant_pressure_abs_kPa = coolant_pressure_g_kPa + 101.325

    st.subheader("External air side")
    air_in_C       = st.number_input("Air inlet temperature (°C)", value=45.0, step=0.1)
    RH_frac        = st.number_input("Relative humidity (fraction)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    face_velocity  = st.number_input("Face velocity (m/s)", min_value=0.1, value=8.0, step=0.1)

    st.subheader("Duty")
    thermal_mode = st.radio("Thermal requirement mode", ["Given required heat rejection", "Given target tube-side outlet"], index=0)
    if thermal_mode == "Given required heat rejection":
        Q_required_input_kW = st.number_input("Required heat rejection (kW)", min_value=0.1, value=408.0, step=1.0)
        T_cool_out_target_input_C = st.number_input(f"{tube_side_label} outlet target (display only, °C)", value=80.0, step=0.1,
                                                    help="In this mode the app derives the required tube-side outlet temperature from the entered duty and flow.")
    else:
        T_cool_out_target_input_C = st.number_input(f"{tube_side_label} outlet target (°C)", value=80.0, step=0.1)
        Q_required_input_kW = st.number_input("Required heat rejection (derived/display, kW)", min_value=0.0, value=408.0, step=1.0,
                                              help="In this mode the app derives the required heat rejection from the tube-side flow and target outlet temperature.")

    st.subheader("Coolant pass layout")
    n_passes = st.number_input("Number of coolant passes", min_value=1, value=3, step=1)
    pass_mode = st.radio("Pass tube distribution mode", ["Equal tubes per pass", "Unequal pass widths"])
    total_tubes_per_row = count_tubes_per_row(core_width_mm, pitch_w_mm, od_w_mm)

    if n_passes == 1:
        tubes_per_row_pass = [total_tubes_per_row]
        pass_widths_mm = [core_width_mm]
    else:
        if pass_mode == "Equal tubes per pass":
            tubes_per_row_pass = split_equal_tubes(total_tubes_per_row, n_passes)
            pass_widths_mm = widths_from_tubes(tubes_per_row_pass, core_width_mm)
            st.caption("Equal-tube mode: pass widths are derived from tube allocation.")
            st.write({f"Pass {i+1} width (mm)": round(w,1) for i,w in enumerate(pass_widths_mm)})
            st.write({f"Pass {i+1} tubes/row": int(n) for i,n in enumerate(tubes_per_row_pass)})
        else:
            pass_widths_mm = []
            remaining = core_width_mm
            defaults = core_width_mm / n_passes
            for i in range(n_passes - 1):
                max_allowed = max(1.0, remaining - (n_passes - i - 1)*1.0)
                w = st.number_input(f"Pass {i+1} width (mm)", min_value=1.0, max_value=float(max_allowed), value=float(min(defaults, max_allowed)), step=0.1)
                pass_widths_mm.append(w)
                remaining -= w
            pass_widths_mm.append(remaining)
            st.write({f"Pass {n_passes} width (auto, mm)": round(pass_widths_mm[-1],1)})
            tubes_per_row_pass = tubes_from_widths(pass_widths_mm, total_tubes_per_row, core_width_mm)
            st.write({f"Pass {i+1} tubes/row": int(n) for i,n in enumerate(tubes_per_row_pass)})

    st.subheader("Model options")
    air_htc_model = st.radio("Air-side HTC model", ["Zukauskas + fin enhancement", "Fin-correlation (Colburn j)", "Kays-London flat-tube surface"], index=1)
    enh_factor = st.number_input("Air-side HTC enhancement", min_value=0.10, value=1.0, step=0.01)
    arr_htc_factor = st.number_input("Arrangement HTC factor", min_value=0.10, value=1.0, step=0.01)
    apply_arr_to_j = st.checkbox("Apply arrangement factor to j-mode", value=True)
    Cj = st.number_input("Cj", min_value=0.001, value=0.12, step=0.001)
    mj = st.number_input("mj", min_value=0.0, value=0.30, step=0.01)
    aj = st.number_input("aj", min_value=-5.0, value=0.05, step=0.01)
    bj = st.number_input("bj", min_value=-5.0, value=0.03, step=0.01)
    cj = st.number_input("cj", min_value=-5.0, value=0.08, step=0.01)
    Cf = st.number_input("Cf", min_value=0.001, value=1.5, step=0.01)
    mf = st.number_input("mf", min_value=0.0, value=0.35, step=0.01)
    use_fin_friction = st.checkbox("Use fin friction in Darcy air-ΔP model", value=True)
    dp_model = st.selectbox("Air-side ΔP model", ["K-per-row model", "Darcy/compact friction model"], index=0)
    K_row = st.number_input("K per row", min_value=0.0, value=1.2, step=0.01)
    k_UA = st.number_input("Overall UA multiplier k_UA", min_value=0.10, value=float(st.session_state.get('k_UA_default', 1.0)), step=0.01, key='k_UA')
    k_DP = st.number_input("Overall air ΔP multiplier k_DP", min_value=0.10, value=float(st.session_state.get('k_DP_default', 1.0)), step=0.01, key='k_DP')
    row_model = st.checkbox("Row-by-row model", value=True)
    var_props = st.checkbox("Variable properties by row", value=True)
    iters_per_row = st.number_input("Iterations per row", min_value=1, value=3, step=1)

# --- Kays-London presets ---
KL_SURFACES = {
    "9.68-0.870": {"label":"Inline plain", "Dh_m":0.002997, "sigma":0.697, "alpha_m2_m3":751.3, "ext_frac":0.795},
    "9.68-0.870-R": {"label":"Inline ruffled", "Dh_m":0.002997, "sigma":0.697, "alpha_m2_m3":751.3, "ext_frac":0.795},
    "9.1-0.737-S": {"label":"Staggered plain", "Dh_m":0.003565, "sigma":0.788, "alpha_m2_m3":734.9, "ext_frac":0.813},
    "9.29-0.737-S-R": {"label":"Staggered ruffled", "Dh_m":0.003510, "sigma":0.788, "alpha_m2_m3":885.8, "ext_frac":0.845},
    "11.32-0.737-S-R": {"label":"Staggered ruffled denser fin", "Dh_m":0.003434, "sigma":0.780, "alpha_m2_m3":748.0, "ext_frac":0.814},
}

KL_CURVES = {
    "9.68-0.870": {
        "Re": np.array([500,700,1000,1500,2000,3000,5000,8000,12000], dtype=float),
        "j":  np.array([0.030,0.027,0.024,0.021,0.019,0.0165,0.0140,0.0120,0.0108], dtype=float),
        "f":  np.array([0.115,0.098,0.082,0.068,0.059,0.049,0.040,0.034,0.030], dtype=float),
    },
    "9.68-0.870-R": {
        "Re": np.array([500,700,1000,1500,2000,3000,5000,8000,12000], dtype=float),
        "j":  np.array([0.035,0.031,0.0275,0.0240,0.0215,0.0185,0.0158,0.0137,0.0123], dtype=float),
        "f":  np.array([0.150,0.128,0.107,0.089,0.078,0.065,0.053,0.045,0.040], dtype=float),
    },
    "9.1-0.737-S": {
        "Re": np.array([500,700,1000,1500,2000,3000,5000,8000,12000], dtype=float),
        "j":  np.array([0.032,0.0285,0.0252,0.0222,0.0200,0.0174,0.0148,0.0128,0.0115], dtype=float),
        "f":  np.array([0.105,0.091,0.077,0.064,0.056,0.047,0.0385,0.033,0.029], dtype=float),
    },
    "9.29-0.737-S-R": {
        "Re": np.array([500,700,1000,1500,2000,3000,5000,8000,12000], dtype=float),
        "j":  np.array([0.038,0.0335,0.0295,0.0258,0.0233,0.0200,0.0170,0.0148,0.0132], dtype=float),
        "f":  np.array([0.135,0.118,0.100,0.084,0.074,0.062,0.051,0.044,0.039], dtype=float),
    },
    "11.32-0.737-S-R": {
        "Re": np.array([500,700,1000,1500,2000,3000,5000,8000,12000], dtype=float),
        "j":  np.array([0.040,0.035,0.0305,0.0268,0.0240,0.0207,0.0175,0.0152,0.0136], dtype=float),
        "f":  np.array([0.145,0.126,0.107,0.090,0.079,0.066,0.055,0.047,0.042], dtype=float),
    },
}

def choose_kl_surface(fin_style, arrangement, fin_louvered, fpi):
    if fin_louvered == "Louvered":
        return None
    if fin_style == "Plate fin" and arrangement == "Inline":
        return "9.68-0.870"
    elif fin_style == "Plate fin" and arrangement == "Staggered":
        return "9.1-0.737-S"
    elif fin_style == "Corrugated fin" and arrangement == "Inline":
        return "9.68-0.870-R"
    elif fin_style == "Corrugated fin" and arrangement == "Staggered":
        return "11.32-0.737-S-R" if fpi >= 10.5 else "9.29-0.737-S-R"
    return "9.68-0.870"

def loglog_interp(x, xp, fp):
    x = float(x)
    x = max(min(x, float(np.max(xp))), float(np.min(xp)))
    return float(np.exp(np.interp(np.log(x), np.log(xp), np.log(fp))))

def kl_airside_from_surface(surface_id, m_dot_air_kg_s, A_frontal_m2, rho_air, mu_air, cp_air, Pr_air, core_depth_m, rho_air_out=None, Kc=0.0, Ke=0.0):
    surf = KL_SURFACES[surface_id]
    curve = KL_CURVES[surface_id]
    sigma = surf['sigma']
    Dh = surf['Dh_m']
    alpha = surf['alpha_m2_m3']
    A_min = sigma*A_frontal_m2
    G = m_dot_air_kg_s/max(A_min,1e-12)
    Re = G*Dh/max(mu_air,1e-12)
    j = loglog_interp(Re, curve['Re'], curve['j'])
    h_o = j*G*cp_air/max(Pr_air**(2.0/3.0), 1e-12)
    f = loglog_interp(Re, curve['Re'], curve['f'])
    core_volume = A_frontal_m2*core_depth_m
    A_total_proxy = alpha*core_volume
    At_over_Amin = 4.0*core_depth_m/max(Dh,1e-12)
    if rho_air_out is None:
        rho_air_out = rho_air
    rho_m = 0.5*(rho_air + rho_air_out)
    dp_fric = (G**2/(2.0*max(rho_m,1e-12)))*f*At_over_Amin
    dp_entr = (G**2/(2.0*max(rho_air,1e-12)))*(Kc + max(0.0,1.0-sigma**2))
    dp_exit = (G**2/(2.0*max(rho_air_out,1e-12)))*max(0.0,1.0-sigma**2-Ke)
    dp_air_core = dp_fric + dp_entr + dp_exit
    return {
        'surface_id':surface_id,
        'surface_label':surf['label'],
        'sigma':sigma,
        'Dh_air_m':Dh,
        'alpha_m2_m3':alpha,
        'ext_frac':surf['ext_frac'],
        'A_min_m2':A_min,
        'G_air':G,
        'Re_air':Re,
        'j_air':j,
        'f_air':f,
        'h_o':h_o,
        'A_total_proxy_m2':A_total_proxy,
        'dp_air_core_Pa':dp_air_core,
    }

# -------------------- Core calculations --------------------
core_w = core_width_mm/1000.0
core_h = core_height_mm/1000.0
core_d = core_depth_mm/1000.0
od_d = od_d_mm/1000.0
od_w = od_w_mm/1000.0
tube_thk = tube_thk_mm/1000.0
ro = tube_corner_radius_mm/1000.0
pitch_w = pitch_w_mm/1000.0
pitch_l = pitch_l_mm/1000.0
fin_thk = fin_thk_mm/1000.0
fin_pitch_m = 0.0254/max(fpi, 1e-9)
L_tube = core_w

# Tube geometry (rounded rectangle / obround)
ri = max(0.0, min(ro - tube_thk, 0.5*(max(od_w-2*tube_thk,1e-12))))
id_d = max(od_d - 2*tube_thk, 1e-12)
id_w = max(od_w - 2*tube_thk, 1e-12)
Ai_one = rounded_rect_area(id_d, id_w, max(ri,0.0))
Pi_one = rounded_rect_perimeter(id_d, id_w, max(ri,0.0))
Ao_one = rounded_rect_area(od_d, od_w, ro)
Po_one = rounded_rect_perimeter(od_d, od_w, ro)
Dh_i_plain = 4.0*Ai_one/max(Pi_one, 1e-12)

k_tube = material_k(tube_material)
k_fin = material_k(fin_material)
oil_inputs = {'model': 'grade_library' if (tube_side_service == 'Oil' and oil_use_grade_library) else 'manual', 'family': oil_family, 'grade': oil_grade, 'rho15': oil_density_kg_m3, 'nu40': oil_nu40_cSt, 'nu100': oil_nu100_cSt, 'rho': oil_density_kg_m3, 'cp': oil_cp_J_kgK, 'k': oil_k_W_mK, 'mu_mPas': oil_mu_mPas}

def tube_props_at(T_C: float):
    return get_tube_side_props(tube_side_service, T_C, coolant_pressure_abs_kPa*1000.0,
                               coolant_name=coolant_name, glycol_pct=glycol_pct, oil_inputs=oil_inputs)

tube_props_inlet = tube_props_at(T_cool_in_C)
rho_tube_inlet = tube_props_inlet['rho']
if tube_side_service == 'Charge air / CAC':
    m_dot_cool = float(charge_air_mdot_kg_s)
else:
    m_dot_cool = rho_tube_inlet*(coolant_Vdot_Lps/1000.0)

cp_req = tube_props_at(max(0.5*(T_cool_in_C + (T_cool_out_target_input_C if thermal_mode == "Given target tube-side outlet" else T_cool_in_C)), -100.0))['cp']
if thermal_mode == "Given required heat rejection":
    Q_required_kW = Q_required_input_kW
    T_cool_out_target_C = T_cool_in_C - (Q_required_kW*1000.0)/max(m_dot_cool*cp_req, 1e-12)
else:
    T_cool_out_target_C = T_cool_out_target_input_C
    Q_required_kW = max(m_dot_cool*cp_req*max(T_cool_in_C - T_cool_out_target_C, 0.0), 0.0)/1000.0

cool_props_inlet = tube_props_inlet
rho_cool_inlet = rho_tube_inlet

if tube_side_service == 'Coolant liquid' and coolant_name in ['Water', 'MEG/Water', 'MPG/Water']:
    coolant_sat_kPa = sat_pressure_water_Pa(T_cool_in_C)/1000.0
    if coolant_pressure_abs_kPa <= coolant_sat_kPa + 5.0 and T_cool_in_C > 95.0:
        st.warning(f"Tube-side liquid absolute pressure ({coolant_pressure_abs_kPa:.1f} kPa abs) is close to or below water saturation pressure at inlet temperature (~{coolant_sat_kPa:.1f} kPa abs). The app keeps liquid properties for stability, but the physical design may be close to boiling.")
if T_cool_out_target_C > T_cool_in_C:
    st.warning(f"{tube_side_label} outlet target is above {tube_side_label.lower()} inlet temperature. Required heat rejection is forced to zero in that case.")

# Internal insert model inside the tube (service-aware approximation)
internal_fin_thk = internal_fin_thk_mm/1000.0
has_internal_fins = (internal_fins_in_tube == 'Yes')
internal_fin_pitch_m = 0.0254/max(internal_fin_fpi, 1e-9) if has_internal_fins else float('inf')
internal_fin_count_per_tube = max(1, int(round((id_w*1000.0/25.4)*internal_fin_fpi))) if has_internal_fins else 0

internal_model = build_internal_insert_model(
    has_internal_fins=has_internal_fins,
    insert_type=internal_insert_type,
    id_w=id_w,
    id_d=id_d,
    tube_length_m=L_tube,
    fin_count_per_tube=internal_fin_count_per_tube,
    fin_thickness_m=internal_fin_thk,
    Ai_plain=Ai_one,
    Pi_plain=Pi_one,
    fin_fpi=internal_fin_fpi,
)
blockage_frac = float(internal_model['blockage_frac'])
Ai_flow_one = float(internal_model['Ai_flow_one'])
Pi_heat_one = float(internal_model['Pi_heat_one'])
A_internal_fin_one_tube = float(internal_model['A_internal_added_one_tube'])
internal_h_enhancement = float(internal_model['internal_h_enhancement'])
internal_dp_multiplier = float(internal_model['internal_dp_multiplier'])
internal_area_ratio_geom = float(internal_model['internal_area_ratio_geom'])
internal_area_ratio = float(internal_model['internal_area_ratio_equiv'])
internal_insert_type_resolved = str(internal_model['insert_type_resolved'])
internal_wetted_area_credit_factor = float(internal_model['wetted_area_credit_factor'])

Dh_i = 4.0*Ai_flow_one/max(Pi_heat_one, 1e-12)
A_i_heat_one = Pi_heat_one*L_tube


# Fin count is based on the core-height dimension for this radiator layout.
# This matches the user convention: tubes-per-row are counted across core width,
# while plate/corrugated fins are stacked along core height at the selected FPI.
N_fins_total = max(1, int(math.floor((core_height_mm/25.4)*fpi + 0.5)))
A_frontal = core_w*core_h
total_tubes = total_tubes_per_row*n_rows

# Geometric free area ratio (very approximate, kept from earlier model)
if fin_style == 'Plate fin':
    free_depth = max(core_d - N_fins_total*fin_thk, 0.01*core_d)
    sigma_geom = free_depth/max(core_d, 1e-12)
else:
    sigma_geom = max(0.10, 1.0 - fin_thk/max(fin_pitch_m,1e-12))

ap_in = air_properties(air_in_C, RH_frac)
rho_air_in = ap_in['rho']
m_dot_air = rho_air_in*face_velocity*A_frontal
vmax_geom = tube_bank_velocity(face_velocity, sigma_geom)

# Inlet-side thermal capacity rates and theoretical limit (engineering guidance)
C_air_inlet_W_K = m_dot_air*ap_in['cp'] if 'ap_in' in locals() else 0.0
C_cool_inlet_W_K = m_dot_cool*cool_props_inlet['cp']
C_min_inlet_W_K = min(C_air_inlet_W_K, C_cool_inlet_W_K)
C_max_inlet_W_K = max(C_air_inlet_W_K, C_cool_inlet_W_K)
Cr_inlet = C_min_inlet_W_K/max(C_max_inlet_W_K, 1e-12)
Q_theoretical_max_kW = max(C_min_inlet_W_K*max(T_cool_in_C - air_in_C, 0.0), 0.0)/1000.0
limiting_side = 'Air side' if C_air_inlet_W_K <= C_cool_inlet_W_K else f'{tube_side_label} side'
face_velocity_req_ideal = (Q_required_kW*1000.0)/(max(ap_in['cp']*rho_air_in*A_frontal*max(T_cool_in_C - air_in_C, 0.0), 1e-12)) if max(T_cool_in_C - air_in_C, 0.0) > 0 else float('nan')
m_dot_air_req_ideal = (Q_required_kW*1000.0)/(max(ap_in['cp']*max(T_cool_in_C - air_in_C, 0.0), 1e-12)) if max(T_cool_in_C - air_in_C, 0.0) > 0 else float('nan')

A_tube_ext_total = Po_one*L_tube*total_tubes

# One fin gross area is based on tube length x core depth (both air-side faces are added later).
# Net plate-fin area subtracts the projected tube-hole area from that gross sheet area.
A_fin_one_gross_single_face_total = L_tube*core_d
A_fin_one_gross_both_faces_total = 2.0*A_fin_one_gross_single_face_total

A_fin_total = 0.0
A_fin_one_airside_single_face_total = 0.0
pass_area_rows = []
for i, (w_pass_mm, tubes_pass_row) in enumerate(zip(pass_widths_mm, tubes_per_row_pass), start=1):
    w_pass = w_pass_mm/1000.0
    N_tubes_pass = tubes_pass_row*n_rows
    A_tube_pass = Po_one*L_tube*N_tubes_pass
    gross_fraction = w_pass / max(core_w, 1e-12)
    if fin_style == 'Plate fin':
        A_fin_one_gross_single_face_pass = gross_fraction * A_fin_one_gross_single_face_total
        A_fin_one_single_face_pass = max(0.0, A_fin_one_gross_single_face_pass - N_tubes_pass*Ao_one)
        A_fin_pass = 2.0*N_fins_total*A_fin_one_single_face_pass
    else:
        waviness = 1.15
        A_fin_one_gross_single_face_pass = gross_fraction * A_fin_one_gross_single_face_total * waviness
        A_fin_one_single_face_pass = A_fin_one_gross_single_face_pass
        A_fin_pass = 2.0*N_fins_total*A_fin_one_single_face_pass
    A_fin_one_airside_single_face_total += A_fin_one_single_face_pass
    A_fin_total += A_fin_pass
    pass_area_rows.append({
        'pass_num': i,
        'pass_width_m': w_pass,
        'tubes_per_row_pass': tubes_pass_row,
        'tubes_in_pass_total': N_tubes_pass,
        'A_tube_ext_pass_m2': A_tube_pass,
        'A_fin_pass_m2': A_fin_pass,
        'A_fin_one_single_face_pass_m2': A_fin_one_single_face_pass,
        'A_fin_one_gross_single_face_pass_m2': A_fin_one_gross_single_face_pass,
    })

# Reference air-side values for comparison
Re_air_ref = reynolds(ap_in['rho'], vmax_geom, max(od_d,1e-12), ap_in['mu'])
Nu_zuk_ref, C2_rows = zukauskas(Re_air_ref, ap_in['Pr'], n_rows=n_rows)
h_o_zuk_ref = (Nu_zuk_ref*ap_in['k']/max(od_d,1e-12))*enh_factor*arr_htc_factor

# Fin-channel hydraulic diameter (very rough placeholder from earlier code)
gap = max(fin_pitch_m - fin_thk, 1e-9)
Dh_air_geom = 4.0*(gap*core_d)/max(2*(gap + core_d), 1e-12)
Re_air_fin_ref = reynolds(ap_in['rho'], vmax_geom, max(Dh_air_geom,1e-12), ap_in['mu'])
j_ref = colburn_j_corrugated(Re_air_fin_ref, fin_pitch_m, Dh_air_geom, pitch_l, pitch_w, louver_pitch_mm/1000.0, louver_angle_deg, Cj, mj, aj, bj, cj)
h_o_j_ref = j_ref*ap_in['rho']*vmax_geom*ap_in['cp']/max(ap_in['Pr']**(2/3), 1e-12)
if apply_arr_to_j:
    h_o_j_ref *= arr_htc_factor
f_air_ref = fin_friction_corr(Re_air_fin_ref, Cf, mf)

# Kays-London UI mapping
kl_surface_auto = choose_kl_surface(fin_style, arrangement, fin_louvered, fpi)
if air_htc_model == 'Kays-London flat-tube surface':
    if fin_louvered == 'Louvered':
        st.warning('Kays-London branch here is intended for non-louvered flat-tube continuous-fin surfaces.')
    kl_mode = st.sidebar.radio('Kays-London preset mode', ['Auto from fin style/arrangement', 'Manual preset'], index=0)
    kl_opts = list(KL_SURFACES.keys())
    if kl_mode == 'Manual preset':
        default_idx = kl_opts.index(kl_surface_auto) if kl_surface_auto in kl_opts else 0
        kl_surface_id = st.sidebar.selectbox('Kays-London flat-tube preset', kl_opts, index=default_idx)
    else:
        kl_surface_id = kl_surface_auto if kl_surface_auto is not None else kl_opts[0]
        st.sidebar.caption(f'Auto-selected preset: {kl_surface_id} — {KL_SURFACES[kl_surface_id]["label"]}')
else:
    kl_surface_id = None

# Fin efficiency based on selected reference h_o
if air_htc_model == 'Zukauskas + fin enhancement':
    h_o_ref_for_eta = h_o_zuk_ref
elif air_htc_model == 'Fin-correlation (Colburn j)':
    h_o_ref_for_eta = h_o_j_ref
else:
    kl_ref = kl_airside_from_surface(kl_surface_id, m_dot_air, A_frontal, ap_in['rho'], ap_in['mu'], ap_in['cp'], ap_in['Pr'], core_d)
    h_o_ref_for_eta = kl_ref['h_o']

eta_fin = fin_efficiency_rect(h_o_ref_for_eta, k_fin, fin_thk, 0.5*max(gap,1e-12))
A_eff_total = A_tube_ext_total + joint_effectiveness*eta_fin*A_fin_total
A_ext_geom_total = A_tube_ext_total + A_fin_total

# Clear geometry / area labels for UI / PDF reporting
A_tube_airside_total_m2 = A_tube_ext_total
A_fin_net_airside_total_m2 = A_fin_total
A_airside_total_geom_m2 = A_ext_geom_total
A_airside_effective_m2 = A_eff_total
one_fin_area_gross_airside_single_face_m2 = A_fin_one_gross_single_face_total
one_fin_area_gross_airside_both_faces_m2 = A_fin_one_gross_both_faces_total
one_fin_area_airside_single_face_m2 = A_fin_one_airside_single_face_total
one_fin_area_airside_both_faces_m2 = 2.0*A_fin_one_airside_single_face_total

# Pass-by-pass calculations
pass_summaries = []
row_results = []
Q_total_W = 0.0
dp_air_total_Pa = 0.0
dp_cool_total_Pa = 0.0
T_cool_pass_in = T_cool_in_C
T_air_out_overall_last_C = air_in_C

for i, p in enumerate(pass_area_rows, start=1):
    w_pass = p['pass_width_m']
    A_frontal_pass = w_pass*core_h
    ap_pass_in = air_properties(air_in_C, RH_frac)
    m_dot_air_pass = ap_pass_in['rho']*face_velocity*A_frontal_pass
    N_parallel = max(p['tubes_in_pass_total'], 1)
    rows_in_pass = max(int(n_rows), 1)

    # Coolant in-tube / pass hydraulics based on pass inlet condition
    cpkg_in = tube_props_at(T_cool_pass_in)
    rho_c = cpkg_in['rho']; mu_c = cpkg_in['mu']; k_c = cpkg_in['k']; Pr_c = cpkg_in['Pr']; cp_c = cpkg_in['cp']
    m_dot_per_tube = m_dot_cool/N_parallel
    v_i = m_dot_per_tube/max(rho_c*Ai_flow_one, 1e-12)
    Re_i = reynolds(rho_c, v_i, Dh_i, mu_c)
    Nu_i, coolant_regime = nu_internal_tube(Re_i, Pr_c, Dh_i, max(L_tube, 1e-12))
    h_i_base = Nu_i*k_c/max(Dh_i,1e-12)
    h_i = h_i_base*internal_area_ratio*internal_h_enhancement

    f_i, friction_regime = f_internal_tube(Re_i)
    dp_cool_pass = f_i*(L_tube/max(Dh_i,1e-12))*0.5*rho_c*(v_i**2)*internal_dp_multiplier
    dp_cool_total_Pa += dp_cool_pass

    # Areas split row-by-row for actual marching
    A_tube_row = p['A_tube_ext_pass_m2']/rows_in_pass
    A_fin_row = p['A_fin_pass_m2']/rows_in_pass

    # Non-row model keeps the previous bulk-pass behavior
    if not row_model:
        T_air_pass_in = air_in_C
        ap = air_properties(T_air_pass_in, RH_frac)

        if air_htc_model == 'Zukauskas + fin enhancement':
            Re_air = reynolds(ap['rho'], vmax_geom, max(od_d,1e-12), ap['mu'])
            Nu_o, _ = zukauskas(Re_air, ap['Pr'], n_rows=n_rows)
            h_o = (Nu_o*ap['k']/max(od_d,1e-12))*enh_factor*arr_htc_factor
            j_air = None
            f_air = f_air_ref
            sigma_used = sigma_geom
            Dh_air_used = Dh_air_geom
            A_min_pass = sigma_geom*A_frontal_pass
            G_air = m_dot_air_pass/max(A_min_pass,1e-12)
            if dp_model == 'K-per-row model':
                dp_air_pass = K_row*n_rows*0.5*ap['rho']*(face_velocity**2)*k_DP
            else:
                f_use = f_air if use_fin_friction else friction_blasius(Re_air)
                dp_air_pass = 4.0*f_use*(core_d/max(Dh_air_geom,1e-12))*0.5*ap['rho']*(vmax_geom**2)*k_DP
        elif air_htc_model == 'Fin-correlation (Colburn j)':
            Re_air = reynolds(ap['rho'], vmax_geom, max(Dh_air_geom,1e-12), ap['mu'])
            j_air = colburn_j_corrugated(Re_air, fin_pitch_m, Dh_air_geom, pitch_l, pitch_w, louver_pitch_mm/1000.0, louver_angle_deg, Cj, mj, aj, bj, cj)
            h_o = j_air*ap['rho']*vmax_geom*ap['cp']/max(ap['Pr']**(2/3), 1e-12)
            if apply_arr_to_j:
                h_o *= arr_htc_factor
            f_air = fin_friction_corr(Re_air, Cf, mf)
            sigma_used = sigma_geom
            Dh_air_used = Dh_air_geom
            A_min_pass = sigma_geom*A_frontal_pass
            G_air = m_dot_air_pass/max(A_min_pass,1e-12)
            if dp_model == 'K-per-row model':
                dp_air_pass = K_row*n_rows*0.5*ap['rho']*(face_velocity**2)*k_DP
            else:
                dp_air_pass = 4.0*f_air*(core_d/max(Dh_air_geom,1e-12))*0.5*ap['rho']*(vmax_geom**2)*k_DP
        else:
            kl = kl_airside_from_surface(kl_surface_id, m_dot_air_pass, A_frontal_pass, ap['rho'], ap['mu'], ap['cp'], ap['Pr'], core_d)
            Re_air = kl['Re_air']
            h_o = kl['h_o']
            j_air = kl['j_air']
            f_air = kl['f_air']
            sigma_used = kl['sigma']
            Dh_air_used = kl['Dh_air_m']
            A_min_pass = kl['A_min_m2']
            G_air = kl['G_air']
            dp_air_pass = kl['dp_air_core_Pa']*k_DP

        eta_f_row = fin_efficiency_rect(h_o, k_fin, fin_thk, 0.5*max(gap,1e-12))
        A_eff_pass = p['A_tube_ext_pass_m2'] + joint_effectiveness*eta_f_row*p['A_fin_pass_m2']
        R_air = 1.0/max(h_o,1e-12)
        R_wall = tube_thk_mm/1000.0/max(k_tube,1e-12)
        R_cool = 1.0/max(h_i,1e-12)
        U_inv = R_air + R_wall + R_cool
        U = (1.0/max(U_inv,1e-12))*k_UA
        UA_pass = U*A_eff_pass

        C_air = m_dot_air_pass*ap['cp']
        C_cool = m_dot_cool*cp_c
        C_min = min(C_air, C_cool)
        C_max = max(C_air, C_cool)
        Cr = C_min/max(C_max,1e-12)
        NTU = UA_pass/max(C_min,1e-12)
        eff = crossflow_mixed_unmixed_effectiveness(NTU, Cr)
        Q_pass_W = eff*C_min*max(T_cool_pass_in - T_air_pass_in, 0.0)
        T_cool_pass_out = T_cool_pass_in - Q_pass_W/max(C_cool,1e-12)
        T_air_pass_out = T_air_pass_in + Q_pass_W/max(C_air,1e-12)
        T_air_out_overall_last_C = T_air_pass_out
        Q_total_W += Q_pass_W
        dp_air_total_Pa += dp_air_pass

        pass_summaries.append({
            'pass_num': i,
            'pass_width_mm': w_pass*1000.0,
            'tubes_per_row_pass': p['tubes_per_row_pass'],
            'tubes_total_pass': p['tubes_in_pass_total'],
            'T_cool_in_C': T_cool_pass_in,
            'T_cool_out_C': T_cool_pass_out,
            'T_air_in_C': T_air_pass_in,
            'T_air_out_C': T_air_pass_out,
            'Q_pass_kW': Q_pass_W/1000.0,
            'v_i_pass_m_s': v_i,
            'Re_i_pass': Re_i,
            'Nu_i_pass': Nu_i,
            'coolant_regime_pass': coolant_regime,
            'f_i_pass': f_i,
            'h_i_pass_W_m2K': h_i,
            'Re_air_pass': Re_air,
            'h_o_pass_W_m2K': h_o,
            'j_air_pass': j_air,
            'f_air_pass': f_air,
            'sigma_pass_used': sigma_used,
            'Dh_air_pass_mm': Dh_air_used*1000.0,
            'A_min_pass_m2': A_min_pass,
            'G_air_pass_kg_m2s': G_air,
            'dp_air_pass_Pa': dp_air_pass,
            'dp_cool_pass_kPa': dp_cool_pass/1000.0,
            'UA_pass_W_K': UA_pass,
            'eff_pass': eff,
            'surface_id': kl_surface_id if air_htc_model == 'Kays-London flat-tube surface' else None,
            'surface_label': KL_SURFACES[kl_surface_id]['label'] if air_htc_model == 'Kays-London flat-tube surface' and kl_surface_id in KL_SURFACES else None,
        })

        for r in range(1, rows_in_pass+1):
            row_results.append({
                'pass_num': i,
                'row_in_pass': r,
                'pass_label': f'P{i}',
                'pass_row': f'P{i}-R{r}',
                'pass_width_mm': w_pass*1000.0,
                'tubes_per_row_pass': p['tubes_per_row_pass'],
                'tubes_total_pass': p['tubes_in_pass_total'],
                'T_cool_in_C': T_cool_pass_in,
                'T_cool_out_C': T_cool_pass_out,
                'T_air_in_C': T_air_pass_in,
                'T_air_out_C': T_air_pass_out,
                'Q_row_kW': (Q_pass_W/rows_in_pass)/1000.0,
                'v_i_m_s': v_i,
                'Re_i': Re_i,
                'Nu_i': Nu_i,
                'coolant_regime': coolant_regime,
                'h_i_W_m2K': h_i,
                'Re_air': Re_air,
                'h_o_W_m2K': h_o,
                'eta_f_row': eta_f_row,
                'j_air': j_air,
                'f_air': f_air,
                'sigma_used': sigma_used,
                'Dh_air_mm': Dh_air_used*1000.0,
                'UA_row_W_K': UA_pass/rows_in_pass,
                'NTU_row': NTU/rows_in_pass,
                'eps_row': eff,
                'dp_air_row_Pa': dp_air_pass/rows_in_pass,
                'dp_cool_row_kPa': dp_cool_pass/1000.0,
                'surface_id': kl_surface_id if air_htc_model == 'Kays-London flat-tube surface' else None,
            })
    else:
        # Actual row-by-row marching:
        # air heats sequentially across rows; coolant is split in parallel across rows within a pass and remixed at the outlet header.
        T_air_in_r = air_in_C
        Q_pass_W = 0.0
        dp_air_pass = 0.0
        row_UAs = []
        row_Re_air = []
        row_h_o = []
        row_j = []
        row_f = []
        row_sigma = []
        row_Dh = []
        row_Amin = []
        row_G = []

        tubes_per_row_local = max(p['tubes_per_row_pass'], 1)
        m_dot_cool_row = m_dot_cool * (tubes_per_row_local / max(N_parallel, 1))
        T_cool_row_in = T_cool_pass_in
        row_Tcool_out = []
        row_dp_cool = []
        row_Re_i = []
        row_Nu_i = []
        row_h_i = []
        row_v_i = []
        row_f_i = []
        row_regime_i = []

        for r in range(1, rows_in_pass+1):
            T_air_out_r = T_air_in_r + 1.0
            T_cool_out_r = T_cool_row_in - 1.0

            for _ in range(max(int(iters_per_row), 1)):
                T_air_prop = 0.5*(T_air_in_r + T_air_out_r) if var_props else T_air_in_r
                ap_r = air_properties(T_air_prop, RH_frac)

                T_cool_prop = 0.5*(T_cool_row_in + T_cool_out_r) if var_props else T_cool_row_in
                cpkg_r = tube_props_at(T_cool_prop)
                rho_c_r = cpkg_r['rho']; mu_c_r = cpkg_r['mu']; k_c_r = cpkg_r['k']; Pr_c_r = cpkg_r['Pr']; cp_c_r = cpkg_r['cp']
                m_dot_per_tube_r = m_dot_cool_row/max(tubes_per_row_local, 1)
                v_i_r = m_dot_per_tube_r/max(rho_c_r*Ai_flow_one, 1e-12)
                Re_i_r = reynolds(rho_c_r, v_i_r, Dh_i, mu_c_r)
                Nu_i_r, coolant_regime_r = nu_internal_tube(Re_i_r, Pr_c_r, Dh_i, max(L_tube, 1e-12))
                h_i_base_r = Nu_i_r*k_c_r/max(Dh_i,1e-12)
                h_i_r = h_i_base_r*internal_area_ratio*internal_h_enhancement
                f_i_r, friction_regime_r = f_internal_tube(Re_i_r)
                dp_cool_row = f_i_r*(L_tube/max(Dh_i,1e-12))*0.5*rho_c_r*(v_i_r**2)*internal_dp_multiplier

                if air_htc_model == 'Zukauskas + fin enhancement':
                    Re_air_r = reynolds(ap_r['rho'], vmax_geom, max(od_d,1e-12), ap_r['mu'])
                    Nu_o_r, _ = zukauskas(Re_air_r, ap_r['Pr'], n_rows=n_rows)
                    h_o_r = (Nu_o_r*ap_r['k']/max(od_d,1e-12))*enh_factor*arr_htc_factor
                    j_air_r = None
                    f_air_r = f_air_ref
                    sigma_r = sigma_geom
                    Dh_air_r = Dh_air_geom
                    A_min_r = sigma_r*A_frontal_pass
                    G_air_r = m_dot_air_pass/max(A_min_r,1e-12)
                    if dp_model == 'K-per-row model':
                        dp_air_row = K_row*0.5*ap_r['rho']*(face_velocity**2)*k_DP
                    else:
                        f_use = f_air_r if use_fin_friction else friction_blasius(Re_air_r)
                        dp_air_row = 4.0*f_use*((core_d/rows_in_pass)/max(Dh_air_geom,1e-12))*0.5*ap_r['rho']*(vmax_geom**2)*k_DP
                elif air_htc_model == 'Fin-correlation (Colburn j)':
                    Re_air_r = reynolds(ap_r['rho'], vmax_geom, max(Dh_air_geom,1e-12), ap_r['mu'])
                    j_air_r = colburn_j_corrugated(Re_air_r, fin_pitch_m, Dh_air_geom, pitch_l, pitch_w, louver_pitch_mm/1000.0, louver_angle_deg, Cj, mj, aj, bj, cj)
                    h_o_r = j_air_r*ap_r['rho']*vmax_geom*ap_r['cp']/max(ap_r['Pr']**(2/3), 1e-12)
                    if apply_arr_to_j:
                        h_o_r *= arr_htc_factor
                    f_air_r = fin_friction_corr(Re_air_r, Cf, mf)
                    sigma_r = sigma_geom
                    Dh_air_r = Dh_air_geom
                    A_min_r = sigma_r*A_frontal_pass
                    G_air_r = m_dot_air_pass/max(A_min_r,1e-12)
                    if dp_model == 'K-per-row model':
                        dp_air_row = K_row*0.5*ap_r['rho']*(face_velocity**2)*k_DP
                    else:
                        dp_air_row = 4.0*f_air_r*((core_d/rows_in_pass)/max(Dh_air_geom,1e-12))*0.5*ap_r['rho']*(vmax_geom**2)*k_DP
                else:
                    kl_r = kl_airside_from_surface(kl_surface_id, m_dot_air_pass, A_frontal_pass, ap_r['rho'], ap_r['mu'], ap_r['cp'], ap_r['Pr'], core_d)
                    Re_air_r = kl_r['Re_air']
                    h_o_r = kl_r['h_o']
                    j_air_r = kl_r['j_air']
                    f_air_r = kl_r['f_air']
                    sigma_r = kl_r['sigma']
                    Dh_air_r = kl_r['Dh_air_m']
                    A_min_r = kl_r['A_min_m2']
                    G_air_r = kl_r['G_air']
                    dp_air_row = (kl_r['dp_air_core_Pa']/rows_in_pass)*k_DP

                eta_f_row = fin_efficiency_rect(h_o_r, k_fin, fin_thk, 0.5*max(gap,1e-12))
                A_eff_row = A_tube_row + joint_effectiveness*eta_f_row*A_fin_row

                R_air_r = 1.0/max(h_o_r,1e-12)
                R_wall_r = tube_thk_mm/1000.0/max(k_tube,1e-12)
                R_cool_r = 1.0/max(h_i_r,1e-12)
                U_inv_r = R_air_r + R_wall_r + R_cool_r
                U_r = (1.0/max(U_inv_r,1e-12))*k_UA
                UA_row = U_r*A_eff_row

                C_air_row = m_dot_air_pass*ap_r['cp']
                C_cool_row = m_dot_cool_row*cp_c_r
                C_min_row = min(C_air_row, C_cool_row)
                C_max_row = max(C_air_row, C_cool_row)
                Cr_row = C_min_row/max(C_max_row, 1e-12)
                NTU_row = UA_row/max(C_min_row, 1e-12)
                eps_row = crossflow_mixed_unmixed_effectiveness(NTU_row, Cr_row)
                dTmax_row = max(T_cool_row_in - T_air_in_r, 0.0)
                Q_row_W = eps_row*C_min_row*dTmax_row

                T_air_out_r = T_air_in_r + Q_row_W/max(C_air_row, 1e-12)
                T_cool_out_r = T_cool_row_in - Q_row_W/max(C_cool_row, 1e-12)

            Q_pass_W += Q_row_W
            dp_air_pass += dp_air_row
            T_air_out_overall_last_C = T_air_out_r
            row_UAs.append(UA_row)
            row_Re_air.append(Re_air_r)
            row_h_o.append(h_o_r)
            row_j.append(j_air_r)
            row_f.append(f_air_r)
            row_sigma.append(sigma_r)
            row_Dh.append(Dh_air_r)
            row_Amin.append(A_min_r)
            row_G.append(G_air_r)
            row_Tcool_out.append(T_cool_out_r)
            row_dp_cool.append(dp_cool_row)
            row_Re_i.append(Re_i_r)
            row_Nu_i.append(Nu_i_r)
            row_h_i.append(h_i_r)
            row_v_i.append(v_i_r)
            row_f_i.append(f_i_r)
            row_regime_i.append(coolant_regime_r)

            row_results.append({
                'pass_num': i,
                'row_in_pass': r,
                'pass_label': f'P{i}',
                'pass_row': f'P{i}-R{r}',
                'pass_width_mm': w_pass*1000.0,
                'tubes_per_row_pass': p['tubes_per_row_pass'],
                'tubes_total_pass': p['tubes_in_pass_total'],
                'T_cool_in_C': T_cool_row_in,
                'T_cool_out_C': T_cool_out_r,
                'T_air_in_C': T_air_in_r,
                'T_air_out_C': T_air_out_r,
                'Q_row_kW': Q_row_W/1000.0,
                'v_i_m_s': v_i_r,
                'Re_i': Re_i_r,
                'Nu_i': Nu_i_r,
                'coolant_regime': coolant_regime_r,
                'h_i_W_m2K': h_i_r,
                'Re_air': Re_air_r,
                'h_o_W_m2K': h_o_r,
                'eta_f_row': eta_f_row,
                'j_air': j_air_r,
                'f_air': f_air_r,
                'sigma_used': sigma_r,
                'Dh_air_mm': Dh_air_r*1000.0,
                'UA_row_W_K': UA_row,
                'NTU_row': NTU_row,
                'eps_row': eps_row,
                'dp_air_row_Pa': dp_air_row,
                'dp_cool_row_kPa': dp_cool_row/1000.0,
                'surface_id': kl_surface_id if air_htc_model == 'Kays-London flat-tube surface' else None,
            })

            T_air_in_r = T_air_out_r

        T_cool_pass_out = float(np.mean(row_Tcool_out)) if row_Tcool_out else T_cool_pass_in
        dp_cool_pass = float(np.mean(row_dp_cool)) if row_dp_cool else 0.0
        dp_cool_total_Pa += dp_cool_pass
        T_air_pass_in = air_in_C
        T_air_pass_out = T_air_in_r

        Q_total_W += Q_pass_W
        dp_air_total_Pa += dp_air_pass

        C_air_pass_in = m_dot_air_pass*ap_pass_in['cp']
        pass_in_props = tube_props_at(T_cool_pass_in)
        C_cool_pass_in = m_dot_cool*pass_in_props['cp']
        eff_pass = Q_pass_W/max(min(C_air_pass_in, C_cool_pass_in)*max(T_cool_pass_in - air_in_C, 0.0), 1e-12)

        pass_summaries.append({
            'pass_num': i,
            'pass_width_mm': w_pass*1000.0,
            'tubes_per_row_pass': p['tubes_per_row_pass'],
            'tubes_total_pass': p['tubes_in_pass_total'],
            'T_cool_in_C': T_cool_pass_in,
            'T_cool_out_C': T_cool_pass_out,
            'T_air_in_C': T_air_pass_in,
            'T_air_out_C': T_air_pass_out,
            'Q_pass_kW': Q_pass_W/1000.0,
            'v_i_pass_m_s': float(np.mean(row_v_i)) if row_v_i else float('nan'),
            'Re_i_pass': float(np.mean(row_Re_i)) if row_Re_i else float('nan'),
            'Nu_i_pass': float(np.mean(row_Nu_i)) if row_Nu_i else float('nan'),
            'coolant_regime_pass': row_regime_i[0] if row_regime_i else 'n/a',
            'f_i_pass': float(np.mean(row_f_i)) if row_f_i else float('nan'),
            'h_i_pass_W_m2K': float(np.mean(row_h_i)) if row_h_i else float('nan'),
            'Re_air_pass': float(np.mean(row_Re_air)) if row_Re_air else float('nan'),
            'h_o_pass_W_m2K': float(np.mean(row_h_o)) if row_h_o else float('nan'),
            'j_air_pass': float(np.mean([x for x in row_j if x is not None])) if any(x is not None for x in row_j) else None,
            'f_air_pass': float(np.mean([x for x in row_f if x is not None])) if any(x is not None for x in row_f) else None,
            'sigma_pass_used': float(np.mean(row_sigma)) if row_sigma else float('nan'),
            'Dh_air_pass_mm': (float(np.mean(row_Dh))*1000.0) if row_Dh else float('nan'),
            'A_min_pass_m2': float(np.mean(row_Amin)) if row_Amin else float('nan'),
            'G_air_pass_kg_m2s': float(np.mean(row_G)) if row_G else float('nan'),
            'dp_air_pass_Pa': dp_air_pass,
            'dp_cool_pass_kPa': dp_cool_pass/1000.0,
            'UA_pass_W_K': float(np.sum(row_UAs)),
            'eff_pass': eff_pass,
            'surface_id': kl_surface_id if air_htc_model == 'Kays-London flat-tube surface' else None,
            'surface_label': KL_SURFACES[kl_surface_id]['label'] if air_htc_model == 'Kays-London flat-tube surface' and kl_surface_id in KL_SURFACES else None,
        })

    T_cool_pass_in = T_cool_pass_out

Q_achieved_kW = Q_total_W/1000.0
T_cool_out_model_C = T_cool_pass_in
T_air_out_model_C = T_air_out_overall_last_C
duty_gap_kW = Q_achieved_kW - Q_required_kW
duty_balance_label = 'Duty excess (kW)' if duty_gap_kW >= 0 else 'Duty shortfall (kW)'
duty_balance_value_kW = abs(duty_gap_kW)
overall_effectiveness_vs_inlet_limit = Q_achieved_kW/max(Q_theoretical_max_kW, 1e-12) if Q_theoretical_max_kW > 0 else float('nan')

report_warnings = []
tube_dp_total_kPa = dp_cool_total_Pa/1000.0
first_pass = pass_summaries[0] if pass_summaries else {}
first_pass_v_i = float(first_pass.get('v_i_pass_m_s', float('nan'))) if first_pass else float('nan')
if tube_side_service == 'Charge air / CAC':
    if tube_dp_total_kPa >= coolant_pressure_abs_kPa:
        report_warnings.append(f'Nonphysical tube-side pressure drop: predicted ΔP_tube = {tube_dp_total_kPa:.1f} kPa exceeds tube absolute inlet pressure = {coolant_pressure_abs_kPa:.1f} kPa. Review charge-air friction model, internal insert assumptions, or input flow/geometry.')
    elif tube_dp_total_kPa >= 0.5*coolant_pressure_abs_kPa:
        report_warnings.append(f'Very high charge-air pressure drop: predicted ΔP_tube = {tube_dp_total_kPa:.1f} kPa is more than 50% of the absolute inlet pressure = {coolant_pressure_abs_kPa:.1f} kPa.')
    if np.isfinite(first_pass_v_i) and first_pass_v_i > 100.0:
        report_warnings.append(f'Very high tube-side gas velocity: pass-1 velocity = {first_pass_v_i:.1f} m/s. Compressibility and insert-loss assumptions should be reviewed carefully.')
if Q_required_kW > Q_theoretical_max_kW + 1e-9:
    report_warnings.append(f'Required duty ({Q_required_kW:.2f} kW) exceeds the ideal inlet thermal limit ({Q_theoretical_max_kW:.2f} kW) for the current inlet conditions.')

# -------------------- Results --------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric('Q achieved (kW)', f'{Q_achieved_kW:.2f}', delta=f'{duty_balance_value_kW:.2f} {"excess" if duty_gap_kW >= 0 else "shortfall"}')
col2.metric('Required heat rejection (kW)', f'{Q_required_kW:.2f}')
col3.metric(duty_balance_label, f'{duty_balance_value_kW:.2f}')
col4.metric(f'{tube_side_label} out model (°C)', f'{T_cool_out_model_C:.2f}', delta=f'{T_cool_out_model_C-T_cool_out_target_C:.2f} vs target')

col5, col6, col7, col8 = st.columns(4)
col5.metric(f'{tube_side_label} out target / required (°C)', f'{T_cool_out_target_C:.2f}')
col6.metric('Air out model (°C)', f'{T_air_out_model_C:.2f}')
col7.metric('Air ΔP total (Pa)', f'{dp_air_total_Pa:.1f}')
col8.metric(f'{tube_side_label} ΔP total (kPa)', f'{dp_cool_total_Pa/1000.0:.3f}')

col9, col10, col11, col12 = st.columns(4)
col9.metric(f'{tube_side_label} velocity, pass 1 (m/s)', f"{first_pass.get('v_i_pass_m_s', float('nan')):.3f}")
col10.metric(f'{tube_side_label} Reynolds, pass 1', f"{first_pass.get('Re_i_pass', float('nan')):.0f}")
col11.metric('Air Reynolds, pass 1', f"{first_pass.get('Re_air_pass', float('nan')):.0f}")
col12.metric('Fin efficiency ηf', f'{eta_fin:.3f}')
col12.caption(f'Bond-adjusted fin factor = {joint_effectiveness*eta_fin:.3f}')

col13a, col14a = st.columns(2)
col13a.metric(f'{tube_side_label} Nu, pass 1', f"{first_pass.get('Nu_i_pass', float('nan')):.2f}")
col14a.metric(f'{tube_side_label} regime / correlation', str(first_pass.get('coolant_regime_pass', 'n/a')))

st.subheader('Air-side area summary')
area1, area2, area3, area4 = st.columns(4)
area1.metric('Total tube area, air side (m²)', f'{A_tube_airside_total_m2:.3f}')
area2.metric('Total net fin area, air side (m²)', f'{A_fin_net_airside_total_m2:.3f}')
area3.metric('Total heat transfer area, air side (m²)', f'{A_airside_total_geom_m2:.3f}')
area4.metric('Effective area, ηf-weighted (m²)', f'{A_airside_effective_m2:.3f}')
st.caption('Air-side total heat transfer area = total external tube area + total net fin area. Effective area additionally applies fin efficiency to the fin contribution.')

st.subheader('Tube and fin count summary')
count1, count2, count3, count4 = st.columns(4)
count1.metric('Tubes per row', f'{int(total_tubes_per_row)}')
count2.metric('Total tubes', f'{int(total_tubes)}')
count3.metric('One fin gross area, both faces (m²)', f'{one_fin_area_gross_airside_both_faces_m2:.4f}')
count4.metric('Total number of fins', f'{int(N_fins_total)}')

count5, count6 = st.columns(2)
count5.metric('One fin net area, both faces (m²)', f'{one_fin_area_airside_both_faces_m2:.4f}')
count6.metric('One fin net area, single face (m²)', f'{one_fin_area_airside_single_face_m2:.4f}')
st.caption('Fin count is based on the core-height / fin-stacking dimension used by the model. One-fin gross area uses tube length x core depth x 2 faces. For plate fin, the net fin area subtracts tube-hole area from that gross sheet area.')

st.subheader('Materials, joints & tube-side internals')
mat1, mat2, mat3, mat4 = st.columns(4)
mat1.metric('Tube material', tube_material)
mat2.metric('Fin material', fin_material)
mat3.metric('Resolved joint type', resolved_joint_type_default)
mat4.metric('Tube-side fluid', tube_side_label)
mat5, mat6, mat7, mat8 = st.columns(4)
mat5.metric('Tube conductivity k (W/m·K)', f'{k_tube:.1f}')
mat6.metric('Fin conductivity k (W/m·K)', f'{k_fin:.1f}')
mat7.metric('Bond effectiveness (-)', f'{joint_effectiveness:.2f}')
mat8.metric('Internal insert type', internal_insert_type_resolved)
mat9, mat10, mat11, mat12 = st.columns(4)
mat9.metric('Internal insert count / tube', f'{int(internal_fin_count_per_tube)}')
mat10.metric('Credited internal added area / tube (m²)', f'{A_internal_fin_one_tube:.4f}')
mat11.metric('Effective internal flow area / tube (mm²)', f'{Ai_flow_one*1e6:.2f}')
mat12.metric('Effective internal Dh (mm)', f'{Dh_i*1000.0:.3f}')
mat13, mat14, mat15, mat16 = st.columns(4)
mat13.metric('Internal equivalent area ratio', f'{internal_area_ratio:.2f}')
mat14.metric('Internal HTC multiplier', f'{internal_h_enhancement:.2f}')
mat15.metric('Internal ΔP multiplier', f'{internal_dp_multiplier:.2f}')
mat16.metric('Oil grade library', oil_grade if tube_side_service == 'Oil' else 'n/a')

col13, col14, col15, col16 = st.columns(4)
col13.metric('Air thermal capacity C_air (W/K)', f'{C_air_inlet_W_K:.1f}')
col14.metric(f'{tube_side_label} thermal capacity C_tube (W/K)', f'{C_cool_inlet_W_K:.1f}')
col15.metric('C_min / C_max', f'{Cr_inlet:.3f}', delta=limiting_side)
col16.metric('Ideal inlet-limit Qmax (kW)', f'{Q_theoretical_max_kW:.2f}', delta=f'{Q_achieved_kW-Q_theoretical_max_kW:.2f} achieved-vs-limit')

capacity_guidance = pd.DataFrame([{
    'Item': 'Air thermal capacity rate C_air', 'Value': C_air_inlet_W_K, 'Units': 'W/K'
}, {
    'Item': f'{tube_side_label} thermal capacity rate C_tube', 'Value': C_cool_inlet_W_K, 'Units': 'W/K'
}, {
    'Item': 'Capacity ratio C_min/C_max', 'Value': Cr_inlet, 'Units': '-'
}, {
    'Item': 'Limiting side at inlet', 'Value': limiting_side, 'Units': ''
}, {
    'Item': 'Ideal inlet thermal limit Qmax', 'Value': Q_theoretical_max_kW, 'Units': 'kW'
}, {
    'Item': 'Overall effectiveness vs inlet limit', 'Value': overall_effectiveness_vs_inlet_limit, 'Units': '-'
}, {
    'Item': 'Ideal air mass flow to hit required Q', 'Value': m_dot_air_req_ideal, 'Units': 'kg/s'
}, {
    'Item': 'Ideal face velocity to hit required Q', 'Value': face_velocity_req_ideal, 'Units': 'm/s'
}])

if Q_required_kW > Q_theoretical_max_kW + 1e-9:
    st.warning(f"Required duty ({Q_required_kW:.2f} kW) exceeds the ideal inlet thermal limit ({Q_theoretical_max_kW:.2f} kW) for the current air flow and inlet temperatures. Limiting side: {limiting_side}. Increase the limiting-side capacity rate before expecting the target duty.")
elif limiting_side == 'Air side':
    st.info(f"Current inlet capacity rates show the air side is limiting (C_air={C_air_inlet_W_K:.1f} W/K, C_tube={C_cool_inlet_W_K:.1f} W/K). Increasing air flow usually helps more than increasing the tube-side flow from this point.")
else:
    st.info(f"Current inlet capacity rates show the {tube_side_label.lower()} side is limiting (C_air={C_air_inlet_W_K:.1f} W/K, C_tube={C_cool_inlet_W_K:.1f} W/K). Increasing the tube-side flow or tube-side heat capacity rate usually helps more than increasing air flow from this point.")

if report_warnings:
    for warn in report_warnings:
        st.warning(warn)

st.markdown('---')
st.subheader('Pass layout')
pass_layout_text = ', '.join([f"P{p['pass_num']}: {p['pass_width_mm']:.1f} mm / {int(p['tubes_per_row_pass'])} tubes-row" for p in pass_summaries])
st.caption(pass_layout_text)

st.subheader('Geometry & intermediate values')
geom_rows = [
    {'Item':'Core width (mm)','Value':core_width_mm},
    {'Item':'Core height (mm)','Value':core_height_mm},
    {'Item':'Core depth (mm)','Value':core_depth_mm},
    {'Item':'Tube OD depth (mm)','Value':od_d_mm},
    {'Item':'Tube OD width (mm)','Value':od_w_mm},
    {'Item':'Tube wall thickness (mm)','Value':tube_thk_mm},
    {'Item':'Tube outside corner radius (mm)','Value':tube_corner_radius_mm},
    {'Item':'Tube inside corner radius (mm)','Value':max(0.0, ri*1000.0)},
    {'Item':'Tube ID depth (mm)','Value':max((od_d - 2*tube_thk)*1000.0, 0.0)},
    {'Item':'Tube ID width (mm)','Value':max((od_w - 2*tube_thk)*1000.0, 0.0)},
    {'Item':'Tube material','Value':tube_material},
    {'Item':'Fin material','Value':fin_material},
    {'Item':'Resolved fin-to-tube joint type','Value':resolved_joint_type_default},
    {'Item':'Bond effectiveness (-)','Value':joint_effectiveness},
    {'Item':'Tube-side service','Value':tube_side_label},
    {'Item':'Oil family','Value':oil_family if tube_side_service == 'Oil' else ''},
    {'Item':'Oil grade','Value':oil_grade if tube_side_service == 'Oil' else ''},
    {'Item':'Oil property model','Value':'Temperature-dependent grade library' if (tube_side_service == 'Oil' and oil_use_grade_library) else ('Manual constants/overrides' if tube_side_service == 'Oil' else '')},
    {'Item':'A_i one tube plain (mm2)','Value':Ai_one*1e6},
    {'Item':'A_i one tube effective flow (mm2)','Value':Ai_flow_one*1e6},
    {'Item':'P_i one tube plain (mm)','Value':Pi_one*1000.0},
    {'Item':'P_i one tube effective / wetted (mm)','Value':Pi_heat_one*1000.0},
    {'Item':'D_h,i plain (mm)','Value':Dh_i_plain*1000.0},
    {'Item':'D_h,i effective (mm)','Value':Dh_i*1000.0},
    {'Item':'A_o one tube cross-section (mm2)','Value':Ao_one*1e6},
    {'Item':'P_o one tube (mm)','Value':Po_one*1000.0},
    {'Item':'N fins total','Value':N_fins_total},
    {'Item':'Tubes per row total','Value':total_tubes_per_row},
    {'Item':'Total tubes','Value':total_tubes},
    {'Item':'One fin gross area, air side both faces (m2)','Value':one_fin_area_gross_airside_both_faces_m2},
    {'Item':'One fin gross area, air side single face (m2)','Value':one_fin_area_gross_airside_single_face_m2},
    {'Item':'One fin net area, air side both faces (m2)','Value':one_fin_area_airside_both_faces_m2},
    {'Item':'One fin net area, air side single face (m2)','Value':one_fin_area_airside_single_face_m2},
    {'Item':'Total tube area, air side (m2)','Value':A_tube_airside_total_m2},
    {'Item':'Total net fin area, air side (m2)','Value':A_fin_net_airside_total_m2},
    {'Item':'Total heat transfer area, air side (m2)','Value':A_airside_total_geom_m2},
    {'Item':'Effective area eta_f-weighted (m2)','Value':A_airside_effective_m2},
    {'Item':'Geometric sigma','Value':sigma_geom},
    {'Item':'Geometric vmax (m/s)','Value':vmax_geom},
    {'Item':'A_frontal (m2)','Value':A_frontal},
    {'Item':'m_dot_air (kg/s)','Value':m_dot_air},
    {'Item':'Tube-side mass flow (kg/s)','Value':m_dot_cool},
    {'Item':'Oil nu@40C (cSt)','Value':oil_nu40_cSt if tube_side_service == 'Oil' else ''},
    {'Item':'Oil nu@100C (cSt)','Value':oil_nu100_cSt if tube_side_service == 'Oil' else ''},
    {'Item':'Internal insert inside tube','Value':internal_fins_in_tube},
    {'Item':'Internal insert type','Value':internal_insert_type_resolved},
    {'Item':'Internal fin style','Value':internal_fin_style},
    {'Item':'Internal fin FPI','Value':internal_fin_fpi},
    {'Item':'Internal fin estimated count / tube','Value':internal_fin_count_per_tube},
    {'Item':'Credited internal added area / tube (m2)','Value':A_internal_fin_one_tube},
    {'Item':'Internal equivalent area ratio','Value':internal_area_ratio},
    {'Item':'Internal geometric area ratio','Value':internal_area_ratio_geom},
    {'Item':'Internal wetted-area credit factor','Value':internal_wetted_area_credit_factor},
    {'Item':'Internal fin HTC multiplier','Value':internal_h_enhancement},
    {'Item':'Internal fin ΔP multiplier','Value':internal_dp_multiplier},
    {'Item':'Air thermal capacity C_air (W/K)','Value':C_air_inlet_W_K},
    {'Item':f'{tube_side_label} thermal capacity C_tube (W/K)','Value':C_cool_inlet_W_K},
    {'Item':'Capacity ratio C_min/C_max','Value':Cr_inlet},
    {'Item':'Limiting side at inlet','Value':limiting_side},
    {'Item':'Ideal inlet thermal limit Qmax (kW)','Value':Q_theoretical_max_kW},
    {'Item':'Ideal air mass flow for required Q (kg/s)','Value':m_dot_air_req_ideal},
    {'Item':'Ideal face velocity for required Q (m/s)','Value':face_velocity_req_ideal},
    {'Item':f'{tube_side_label} operating pressure (kPa g)','Value':coolant_pressure_g_kPa},
    {'Item':f'{tube_side_label} operating pressure (kPa abs)','Value':coolant_pressure_abs_kPa},
    {'Item':'Thermal mode','Value':thermal_mode},
]
st.dataframe(pd.DataFrame(geom_rows), use_container_width=True, hide_index=True)

st.subheader('Thermal capacity guidance')
st.dataframe(capacity_guidance, use_container_width=True, hide_index=True)

st.subheader('Air-side model comparison')
comp_rows = [
    {'Model':'Zukauskas + enhancement','Re_air':Re_air_ref,'h_o_W_m2K':h_o_zuk_ref,'j':None,'f':None,'sigma_used':sigma_geom,'Dh_air_mm':od_d_mm},
    {'Model':'Generic Colburn-j','Re_air':Re_air_fin_ref,'h_o_W_m2K':h_o_j_ref,'j':j_ref,'f':f_air_ref,'sigma_used':sigma_geom,'Dh_air_mm':Dh_air_geom*1000.0},
]
if air_htc_model == 'Kays-London flat-tube surface' and kl_surface_id is not None:
    kl_ref = kl_airside_from_surface(kl_surface_id, m_dot_air, A_frontal, ap_in['rho'], ap_in['mu'], ap_in['cp'], ap_in['Pr'], core_d)
    comp_rows.append({'Model':f'Kays-London {kl_surface_id}','Re_air':kl_ref['Re_air'],'h_o_W_m2K':kl_ref['h_o'],'j':kl_ref['j_air'],'f':kl_ref['f_air'],'sigma_used':kl_ref['sigma'],'Dh_air_mm':kl_ref['Dh_air_m']*1000.0})
st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

if air_htc_model == 'Kays-London flat-tube surface' and kl_surface_id is not None:
    st.info(f"Kays-London surrogate surface actually used: {kl_surface_id} — {KL_SURFACES[kl_surface_id]['label']} | sigma={KL_SURFACES[kl_surface_id]['sigma']:.3f}, Dh={KL_SURFACES[kl_surface_id]['Dh_m']*1000.0:.3f} mm")

st.subheader(f'{tube_side_label} pass summary')
pass_df = pd.DataFrame(pass_summaries)
st.dataframe(pass_df, use_container_width=True, hide_index=True)

st.subheader('Row-wise intermediate results')
st.caption(f'With row-by-row mode ON, air temperature marches sequentially through each row and {tube_side_label.lower()} is split in parallel across rows within a pass, then remixed at the outlet header.')
rows_df = pd.DataFrame(row_results)
row_filter = st.selectbox('Show rows for', ['All passes'] + [f'P{i}' for i in range(1, int(n_passes)+1)], index=0)
if row_filter == 'All passes':
    st.dataframe(rows_df, use_container_width=True, hide_index=True)
else:
    st.dataframe(rows_df[rows_df['pass_label']==row_filter], use_container_width=True, hide_index=True)

# -------------------- Downloads --------------------
def make_pdf_report_bytes(summary_df, pass_df, rows_df, geom_rows):
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors

    tube_temp_label = f"{tube_side_label} temperature"
    tube_dp_label = f"{tube_side_label} ΔP"

    def fmt(v, nd=3):
        if v is None or v == '' or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return '—'
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{nd}f}"
        return str(v)

    def grouped_geom_dict(rows):
        return {str(r['Item']): r['Value'] for r in rows}

    gd = grouped_geom_dict(geom_rows)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    left = 14*mm
    right = W - 14*mm
    y = H - 16*mm

    def new_page(title=None):
        nonlocal y
        c.showPage()
        y = H - 16*mm
        if title:
            section_title(title)

    def line(txt='', size=9, bold=False, gap=4.5, color=colors.black):
        nonlocal y
        if y < 16*mm:
            new_page()
        c.setFillColor(color)
        c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
        c.drawString(left, y, str(txt))
        y -= gap*mm
        c.setFillColor(colors.black)

    def wrapped(txt, size=9, bold=False, gap=4.3):
        nonlocal y
        c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
        max_chars = 115 if size >= 9 else 135
        words = str(txt).split()
        cur = ''
        for w in words:
            test = (cur + ' ' + w).strip()
            if len(test) <= max_chars:
                cur = test
            else:
                line(cur, size=size, bold=bold, gap=gap)
                cur = w
        if cur:
            line(cur, size=size, bold=bold, gap=gap)

    def section_title(txt):
        line(txt, size=11, bold=True, gap=5.5)
        c.setStrokeColor(colors.HexColor('#888888'))
        c.line(left, y+1.5*mm, right, y+1.5*mm)
        c.setStrokeColor(colors.black)
        y_shift = 2.5
        nonlocal_y = None
        # use closure variable directly
        globals()

    def kv(label, value, units=''):
        suffix = f' {units}' if units else ''
        wrapped(f'{label}: {value}{suffix}', size=9, gap=4.0)

    # Title
    line('Radiator / CAC / Oil Cooler Thermal Report', size=14, bold=True, gap=5.2)
    line(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), size=8, gap=4.5)

    # Executive summary
    section_title('1. Executive summary')
    kv('Thermal mode', thermal_mode)
    kv('Tube-side service', tube_side_label)
    kv('Required duty', fmt(Q_required_kW, 2), 'kW')
    kv('Achieved duty', fmt(Q_achieved_kW, 2), 'kW')
    kv(duty_balance_label, fmt(duty_balance_value_kW, 2), 'kW')
    kv('Tube-side inlet temperature', fmt(T_cool_in_C, 2), '°C')
    kv('Tube-side target outlet temperature', fmt(T_cool_out_target_C, 2), '°C')
    kv('Tube-side model outlet temperature', fmt(T_cool_out_model_C, 2), '°C')
    kv('Air inlet temperature', fmt(air_in_C, 2), '°C')
    kv('Air model outlet temperature', fmt(T_air_out_model_C, 2), '°C')
    kv('Air total pressure drop', fmt(dp_air_total_Pa, 1), 'Pa')
    kv(f'{tube_side_label} total pressure drop', fmt(tube_dp_total_kPa, 3), 'kPa')
    if tube_side_service in ['Charge air / CAC', 'Oil']:
        kv('Tube-side inlet pressure, gauge', fmt(coolant_pressure_g_kPa, 3), 'kPa')
        kv('Tube-side inlet pressure, absolute', fmt(coolant_pressure_abs_kPa, 3), 'kPa')

    section_title('2. Warnings / sanity checks')
    if report_warnings:
        for w in report_warnings:
            wrapped('• ' + w, size=9, gap=4.0)
    else:
        wrapped('No immediate report sanity flags were triggered for this case.', size=9, gap=4.0)

    section_title('3. Geometry, counts, areas')
    kv('Core width', fmt(core_width_mm, 1), 'mm')
    kv('Core height', fmt(core_height_mm, 1), 'mm')
    kv('Core depth', fmt(core_depth_mm, 1), 'mm')
    kv('Tubes per row', int(total_tubes_per_row))
    kv('Total tubes', int(total_tubes))
    kv('Total number of fins', int(N_fins_total))
    kv('One fin gross area, both faces', fmt(one_fin_area_gross_airside_both_faces_m2, 6), 'm²')
    kv('One fin net area, both faces', fmt(one_fin_area_airside_both_faces_m2, 6), 'm²')
    kv('Total tube area, air side', fmt(A_tube_airside_total_m2, 3), 'm²')
    kv('Total net fin area, air side', fmt(A_fin_net_airside_total_m2, 3), 'm²')
    kv('Total geometric heat-transfer area, air side', fmt(A_airside_total_geom_m2, 3), 'm²')
    kv('Effective air-side area, ηf-weighted', fmt(A_airside_effective_m2, 3), 'm²')
    kv('Fin efficiency ηf', fmt(eta_fin, 6))

    section_title('4. Materials, joint, and tube-side insert')
    kv('Tube material', tube_material)
    kv('Fin material', fin_material)
    kv('Resolved fin-to-tube joint type', resolved_joint_type_default)
    kv('Bond effectiveness', fmt(joint_effectiveness, 3))
    kv('Internal insert present', 'Yes' if internal_fins_in_tube else 'No')
    kv('Internal insert type', internal_insert_type_resolved)
    kv('Internal fin style', internal_fin_style)
    kv('Internal fin FPI', fmt(internal_fin_fpi, 2))
    kv('Internal insert count per tube', int(internal_fin_count_per_tube))
    kv('Effective internal flow area per tube', fmt(Ai_flow_one*1e6, 3), 'mm²')
    kv('Effective internal hydraulic diameter', fmt(Dh_i*1000.0, 3), 'mm')
    kv('Internal equivalent area ratio', fmt(internal_area_ratio, 3))
    kv('Internal HTC multiplier', fmt(internal_h_enhancement, 3))
    kv('Internal ΔP multiplier', fmt(internal_dp_multiplier, 3))
    if tube_side_service == 'Oil':
        kv('Oil family', oil_family)
        kv('Oil grade', oil_grade)
        kv('Oil property model', 'Temperature-dependent grade library' if oil_use_grade_library else 'Manual constants / override')
        kv('Oil ν@40°C', fmt(oil_nu40_cSt, 3), 'cSt')
        kv('Oil ν@100°C', fmt(oil_nu100_cSt, 3), 'cSt')

    new_page('5. Capacity guidance')
    kv('Air thermal capacity rate C_air', fmt(C_air_inlet_W_K, 1), 'W/K')
    kv(f'{tube_side_label} thermal capacity rate C_tube', fmt(C_cool_inlet_W_K, 1), 'W/K')
    kv('Capacity ratio C_min / C_max', fmt(Cr_inlet, 4))
    kv('Limiting side', limiting_side)
    kv('Ideal inlet thermal limit Qmax', fmt(Q_theoretical_max_kW, 3), 'kW')
    kv('Overall effectiveness vs inlet limit', fmt(overall_effectiveness_vs_inlet_limit, 6))
    kv('Ideal air mass flow for required duty', fmt(m_dot_air_req_ideal, 3), 'kg/s')
    kv('Ideal face velocity for required duty', fmt(face_velocity_req_ideal, 3), 'm/s')
    kv('Pass layout', ', '.join([f"P{int(r['pass_num'])}: {float(r['pass_width_mm']):.1f} mm / {int(r['tubes_per_row_pass'])} tubes-row" for _, r in pass_df.iterrows()]))

    new_page('6. Pass summary')
    wrapped('Each coolant / tube-side pass is shown below. For charge-air service, the pass labels still follow the tube-side path through the core.', size=8.5, gap=4.0)
    for _, r in pass_df.iterrows():
        line(f"Pass P{int(r['pass_num'])}", size=9.5, bold=True, gap=4.5)
        wrapped(
            f"Width {float(r['pass_width_mm']):.1f} mm | Tubes/row {int(r['tubes_per_row_pass'])} | Total tubes {int(r['tubes_total_pass'])} | "
            f"{tube_side_label} {float(r['T_cool_in_C']):.2f} → {float(r['T_cool_out_C']):.2f} °C | Air {float(r['T_air_in_C']):.2f} → {float(r['T_air_out_C']):.2f} °C",
            size=8.5, gap=3.7
        )
        wrapped(
            f"Q_pass {float(r['Q_pass_kW']):.3f} kW | UA {float(r['UA_pass_W_K']):.1f} W/K | ε {float(r['eff_pass']):.4f} | "
            f"v_tube {float(r['v_i_pass_m_s']):.3f} m/s | Re_tube {float(r['Re_i_pass']):.0f} | Nu_tube {float(r['Nu_i_pass']):.2f} | h_tube {float(r['h_i_pass_W_m2K']):.1f} W/m²K",
            size=8.5, gap=3.7
        )
        wrapped(
            f"Re_air {float(r['Re_air_pass']):.0f} | h_air {float(r['h_o_pass_W_m2K']):.1f} W/m²K | ΔP_air {float(r['dp_air_pass_Pa']):.2f} Pa | "
            f"ΔP_tube {float(r['dp_cool_pass_kPa']):.3f} kPa | Regime {r['coolant_regime_pass']}",
            size=8.5, gap=4.2
        )
        line('', size=8, gap=1.5)

    new_page('7. Row-by-row appendix')
    wrapped('This appendix shows the row-marching details. For parallel tube-side branches inside a pass, the same branch inlet temperature is shown for each row and the branch outlet temperature changes with local air heating.', size=8.5, gap=4.0)
    current_pass = None
    for _, r in rows_df.iterrows():
        pass_no = int(r['pass_num'])
        if current_pass != pass_no:
            current_pass = pass_no
            line(f'Pass P{pass_no}', size=9, bold=True, gap=4.2)
        wrapped(
            f"{r['pass_row']}: Air {float(r['T_air_in_C']):.2f} → {float(r['T_air_out_C']):.2f} °C | "
            f"{tube_side_label} {float(r['T_cool_in_C']):.2f} → {float(r['T_cool_out_C']):.2f} °C | Q_row {float(r['Q_row_kW']):.3f} kW | "
            f"Re_tube {float(r['Re_i']):.0f} | Nu_tube {float(r['Nu_i']):.2f} | ΔP_tube {float(r['dp_cool_row_kPa']):.3f} kPa | ΔP_air {float(r['dp_air_row_Pa']):.2f} Pa",
            size=7.5, gap=3.4
        )

    new_page('8. Methods / assumptions')
    methods = [
        'Tube-side geometry uses a rounded-rectangle / obround tube model.',
        'Air-side total area = total exposed tube area + total net fin area. Effective area applies fin efficiency and bond effectiveness to the fin contribution.',
        'Tube-side multipass logic: passes in series, tubes within each pass in parallel.',
        f'With row marching enabled, air is marched row by row and the {tube_side_label.lower()} branches in a pass are remixed at the outlet header.',
        'Tube-side in-tube correlation: laminar/developing below Re≈2300, Gnielinski above Re≈4000, with transition blending in between.',
        'This report uses the current app friction and heat-transfer model exactly as executed for the case shown. For charge-air cases with very high predicted velocity or pressure drop, the result should be checked against compressible-flow reality and test data.'
    ]
    for m in methods:
        wrapped('• ' + m, size=8.5, gap=4.0)

    c.save()
    buf.seek(0)
    return buf.getvalue()

summary_df = pd.DataFrame([{
    'thermal_mode': thermal_mode,
    'Q_required_kW': Q_required_kW,
    'Q_achieved_kW': round(Q_achieved_kW,3),
    'duty_balance_label': duty_balance_label,
    'duty_balance_kW': round(duty_balance_value_kW,3),
    'tube_side_service': tube_side_label,
    'report_warnings': ' | '.join(report_warnings),
    'tube_material': tube_material,
    'fin_material': fin_material,
    'oil_family': oil_family if tube_side_service == 'Oil' else '',
    'oil_grade': oil_grade if tube_side_service == 'Oil' else '',
    'oil_property_model': ('grade_library' if (tube_side_service == 'Oil' and oil_use_grade_library) else ('manual' if tube_side_service == 'Oil' else '')),
    'T_tube_in_C': T_cool_in_C,
    'T_tube_out_target_C': T_cool_out_target_C,
    'T_tube_out_model_C': round(T_cool_out_model_C,3),
    'T_air_in_C': air_in_C,
    'tube_pressure_g_kPa': coolant_pressure_g_kPa,
    'tube_pressure_abs_kPa': coolant_pressure_abs_kPa,
    'oil_nu40_cSt': round(oil_nu40_cSt,3) if tube_side_service == 'Oil' else '',
    'oil_nu100_cSt': round(oil_nu100_cSt,3) if tube_side_service == 'Oil' else '',
    'resolved_joint_type': resolved_joint_type_default,
    'bond_effectiveness': round(joint_effectiveness,4),
    'internal_insert_type': internal_insert_type_resolved,
    'internal_area_ratio_equiv': round(internal_area_ratio,4),
    'internal_area_ratio_geom': round(internal_area_ratio_geom,4),
    'T_air_out_model_C': round(T_air_out_model_C,3),
    'dp_air_total_Pa': round(dp_air_total_Pa,3),
    'dp_tube_total_kPa': round(dp_cool_total_Pa/1000.0,3),
    'internal_fins_in_tube': internal_fins_in_tube,
    'internal_fin_style': internal_fin_style,
    'internal_fin_fpi': internal_fin_fpi,
    'internal_fin_count_per_tube': int(internal_fin_count_per_tube),
    'Ai_flow_one_mm2': round(Ai_flow_one*1e6,3),
    'Dh_i_effective_mm': round(Dh_i*1000.0,3),
    'tubes_per_row_total': int(total_tubes_per_row),
    'total_tubes': int(total_tubes),
    'one_fin_gross_area_airside_both_faces_m2': round(one_fin_area_gross_airside_both_faces_m2, 4),
    'one_fin_net_area_airside_both_faces_m2': round(one_fin_area_airside_both_faces_m2, 4),
    'total_number_of_fins': int(N_fins_total),
    'A_tube_airside_total_m2': round(A_tube_airside_total_m2,3),
    'A_fin_net_airside_total_m2': round(A_fin_net_airside_total_m2,3),
    'A_airside_total_geom_m2': round(A_airside_total_geom_m2,3),
    'A_airside_effective_m2': round(A_airside_effective_m2,3),
    'eta_fin': round(eta_fin,6),
    'C_air_inlet_W_K': round(C_air_inlet_W_K,3),
    'C_tube_inlet_W_K': round(C_cool_inlet_W_K,3),
    'C_ratio_min_max': round(Cr_inlet,6),
    'limiting_side': limiting_side,
    'Qmax_inlet_kW': round(Q_theoretical_max_kW,3),
    'overall_effectiveness_vs_inlet_limit': round(overall_effectiveness_vs_inlet_limit,6) if pd.notna(overall_effectiveness_vs_inlet_limit) else None,
    'm_dot_air_req_ideal_kg_s': round(m_dot_air_req_ideal,3) if pd.notna(m_dot_air_req_ideal) else None,
    'face_velocity_req_ideal_m_s': round(face_velocity_req_ideal,3) if pd.notna(face_velocity_req_ideal) else None,
    'n_passes': int(n_passes),
    'pass_mode': pass_mode,
    'pass_layout': pass_layout_text,
    'air_htc_model': air_htc_model,
    'kays_surface_id': kl_surface_id if kl_surface_id else '',
}])

st.download_button('Download outputs.csv', data=summary_df.to_csv(index=False).encode('utf-8'), file_name='outputs.csv', mime='text/csv')
st.download_button('Download passes.csv', data=pass_df.to_csv(index=False).encode('utf-8'), file_name='passes.csv', mime='text/csv')
st.download_button('Download rows.csv', data=rows_df.to_csv(index=False).encode('utf-8'), file_name='rows.csv', mime='text/csv')
st.download_button('Download PDF report', data=make_pdf_report_bytes(summary_df, pass_df, rows_df, geom_rows), file_name='radiator_report.pdf', mime='application/pdf')

# -------------------- Notes --------------------
st.markdown('---')
st.markdown('### Notes')
st.markdown('- **Password**: this app now checks `APP_PASSWORD` from Streamlit secrets at startup.')
st.markdown('- **Tube-side properties**: coolant liquid, charge air, and oil are all supported. Hot liquid coolant uses a liquid-stable property model to avoid false vapour-density glitches. In row-by-row mode, tube-side properties are re-evaluated for each row/branch using the local mean tube-side temperature.')
st.markdown('- **Tube-side in-tube correlation**: laminar/developing below Re≈2300, Gnielinski above Re≈4000, with a transition blend in between.')
st.markdown('- **Air-side models**: Zukauskas, generic Colburn-j, and optional Kays-London surrogate branch are all still present in this version.')
st.markdown('- **Fin efficiency**: fin thickness, selected fin material conductivity, fin pitch/gap, and row pitch are used to compute \(\eta_f\); the fin contribution is then multiplied by the fin-to-tube bond effectiveness factor.')
st.markdown('- **Internal inserts**: CAC internal-fin mode credits more real internal area, while oil-turbulator mode credits less area but stronger mixing/HTC enhancement and stronger ΔP penalty.')
st.markdown('- **Crossflow (tube side mixed / air unmixed):** Kays–London mixed–unmixed $\varepsilon(\mathrm{NTU}, C_r)$. In row-by-row mode, air marches sequentially by row while the tube-side stream is split across rows in parallel, each row branch is solved with its own local properties, and the branches are then remixed at the pass outlet.')
st.markdown('- **Air (K-based):** $\Delta P = K_{\text{tot}}\,(\tfrac{1}{2}\rho V^2)$ with entrance/exit + per-row + misc.')
st.markdown('- **Air (Darcy channel):** $\Delta P = 4f\,(L/D_h)\,(\tfrac{1}{2}\rho V^2)$ (+ header/minors as applicable).')
