#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         toolData.py
# Date:         05.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Libraries of the Theory & Tools pages (typical values from the literature, all editable in the GUI):
    1) MATERIALS:   thermal conductivity k [W/(m K)], density rho [kg/m3] and specific heat cp [J/(kg K)]
    2) PACKAGES:    typical power semiconductor packages and modules: chip, layers inside the package (junction →
                    case), layers outside (case → coolant), cooling, extra paths (housing, pins), second side (DSC)
    3) BOARDS:      board technologies that replace the layers with the role 'board' of a stack
    4) COOLING:     heat transfer coefficients h [W/(m2 K)] of typical cooling methods
    5) DEVICES:     generic loss parameters of Si IGBT/diode, Si MOSFET, SiC MOSFET and GaN HEMT
    6) TESTS:       catalogue of qualification tests with their standard conditions
The values are typical values for orientation (room temperature, datasheet test points); they must be checked for the
specific part, material grade and temperature before they are used for a design decision.

Copied from OPTELIA (RAISE, src/tools) for the Theory & Tools pages of the TherMOS GUI.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import math

#######################################################################################################################
# Materials
#######################################################################################################################
# name: (k [W/(m K)], rho [kg/m3], cp [J/(kg K)], group, note)
MATERIALS = {
    'Si':                  (148.0, 2330, 705, 'Semiconductor', "k falls to ≈ 105 W/(m K) at 100 °C"),
    'SiC (4H)':            (370.0, 3210, 690, 'Semiconductor', "300–490 W/(m K) depending on doping and orientation"),
    'GaN':                 (130.0, 6150, 490, 'Semiconductor', "GaN-on-Si devices conduct the heat through the Si "
                                                               "substrate (use Si for the substrate layer)"),
    'Cu':                  (390.0, 8960, 385, 'Metal', "leadframes, DBC copper, baseplates, inlays"),
    'Al':                  (237.0, 2700, 900, 'Metal', "pure aluminium (wire, metallisation)"),
    'Al alloy (6063)':     (200.0, 2700, 900, 'Metal', "extruded heat sinks"),
    'Al die-cast (A380)':  (96.0, 2710, 963, 'Metal', "housings and cold plates"),
    'AlSiC':               (180.0, 3000, 740, 'Metal', "baseplates matched to ceramics"),
    'SAC305 solder':       (58.0, 7380, 232, 'Joining', "lead-free die attach and board solder"),
    'SnPb solder':         (50.0, 8400, 167, 'Joining', "eutectic tin-lead"),
    'High-Pb solder':      (35.0, 11000, 150, 'Joining', "die attach of discrete packages"),
    'Sintered Ag':         (200.0, 8500, 235, 'Joining', "150–250 W/(m K) depending on porosity"),
    'Al2O3 (96 %)':        (24.0, 3750, 880, 'Ceramic', "standard DBC ceramic"),
    'AlN':                 (170.0, 3260, 740, 'Ceramic', "high-performance DBC/AMB ceramic"),
    'Si3N4':               (90.0, 3200, 680, 'Ceramic', "AMB ceramic, high fracture toughness"),
    'FR4 (through-plane)': (0.3, 1850, 1100, 'Board', "glass-epoxy laminate perpendicular to the layers"),
    'FR4 (in-plane)':      (0.8, 1850, 1100, 'Board', "glass-epoxy laminate along the layers"),
    'IMS dielectric':      (2.0, 2100, 1000, 'Board', "1–10 W/(m K) depending on the filler"),
    'Mold compound (EMC)': (0.9, 1900, 900, 'Package', "epoxy mold compound of packages"),
    'Thermal grease':      (3.0, 2600, 800, 'TIM', "1–8 W/(m K), 25–100 µm bond line"),
    'Gap pad':             (3.0, 3000, 1000, 'TIM', "1–6 W/(m K), 0.5–3 mm, electrically insulating"),
    'Phase change (PCM)':  (5.0, 2500, 1000, 'TIM', "3–8 W/(m K), 25–50 µm after melting"),
    'Insulating foil':     (1.5, 1500, 1100, 'TIM', "polyimide/silicone insulating pad, 0.1–0.25 mm"),
    'Graphite sheet (z)':  (5.0, 1900, 850, 'TIM', "through-plane, in-plane up to 1500 W/(m K)"),
    'Gap filler (dispensed)': (3.5, 3200, 1000, 'TIM', "liquid dispensed gap filler, 2–6 W/(m K), 0.2–2 mm "
                                                         "(estimate)"),
    'TIM pad (high performance)': (6.0, 3200, 1000, 'TIM', "filled silicone pad 5–12 W/(m K) (estimate)"),
    'Solder TIM (indium)': (82.0, 7310, 233, 'TIM', "metallic TIM, indium foil or solder, 50–100 µm"),
    'Polyimide foil (Kapton)': (0.12, 1420, 1090, 'TIM', "electrical insulation foil without filler"),
    'Silicone gel':        (0.2, 970, 1500, 'Package', "module encapsulation gel, 0.15–0.3 W/(m K)"),
    'Potting (silicone)':  (0.5, 1500, 1200, 'Package', "filled silicone potting 0.2–1 W/(m K) (estimate)"),
    'Potting (epoxy, filled)': (1.0, 2000, 1000, 'Package', "filled epoxy potting 0.5–2 W/(m K) (estimate)"),
    'Steel':               (50.0, 7850, 460, 'Metal', "housings, clamps"),
    'Air':                 (0.026, 1.2, 1005, 'Gas', "still air, gaps and voids"),
    'Water-glycol 50/50':  (0.4, 1070, 3300, 'Fluid', "coolant (for reference, convection by h)"),
}
MATERIAL_REFS = ['Incropera2007', 'Wintrich2015']


def material(name):
    """Dictionary of a library material."""
    k, rho, cp, group, note = MATERIALS[name]
    return {'k': k, 'rho': rho, 'cp': cp, 'group': group, 'note': note}


def via_k(pitch_mm, drill_mm, plating_um, filled=False, k_board=0.3, k_cu=390.0, k_fill=0.3):
    """
    Effective through-plane conductivity of a PCB area with a square array of plated thermal vias (parallel
    conduction of the copper barrels, the filling and the laminate):
        k_eff = phi_cu k_Cu + phi_fill k_fill + (1 - phi_cu - phi_fill) k_board
    with the area fractions phi of the copper barrel and the filling of one via cell pitch x pitch.
    """
    r_o, t = drill_mm / 2, plating_um / 1000
    r_i = max(r_o - t, 0.0)
    a_cell = pitch_mm ** 2
    phi_cu = math.pi * (r_o ** 2 - r_i ** 2) / a_cell
    phi_hole = math.pi * r_i ** 2 / a_cell
    k_hole = k_fill if filled else 0.026
    return phi_cu * k_cu + phi_hole * k_hole + (1 - phi_cu - phi_hole) * k_board


#######################################################################################################################
# Layer stacks (packages, boards, cooling)
#######################################################################################################################
# a layer: name, role, material, t [mm], w x l footprint [mm] (0 = unlimited), spreading angle [deg], k, rho, cp
# role: attach | package | board | tim | iso | housing | cooler
def L(name, role, mat, t, w=0.0, l=0.0, ang=45.0, k=None, rho=None, cp=None):
    m = MATERIALS[mat]
    return {'layer': name, 'role': role, 'material': mat, 't [mm]': t, 'w [mm]': w, 'l [mm]': l,
            'angle [°]': ang, 'k [W/mK]': m[0] if k is None else k, 'rho [kg/m³]': m[1] if rho is None else rho,
            'cp [J/kgK]': m[2] if cp is None else cp}


_hs = [L('Heat sink base', 'cooler', 'Al alloy (6063)', 5.0, 100, 100)]
_box = [L('Housing wall', 'housing', 'Al die-cast (A380)', 3.0, 80, 80)]
_pcb_vias = [L('Cu pad (top layer)', 'board', 'Cu', 0.07, 12, 12),
             L('FR4 with thermal vias', 'board', 'FR4 (through-plane)', 1.5, 12, 12,
               k=round(via_k(1.0, 0.3, 25), 2)),
             L('Cu (bottom layer)', 'board', 'Cu', 0.07, 20, 20)]

# Every package: the chip (heat source on its top surface = junction), the layers inside the package from the chip to
# the case (= the cooling area of the datasheet, junction-to-case Z_th,jc), the layers outside the package from the
# case to the coolant, the cooling (h on the outer face of the last outside layer) and the optional extra paths
# (estimates): R_jtop junction → top of the plastic housing (A_top cooled by h_top), R_jpins junction → pins / board
# and R_pins_amb board → ambient. 'second': the other side of a double-side cooled package. 'kind' selects the
# picture.
PACKAGES = {
    'TO-247 on heat sink (insulating pad)': {
        'kind': 'tht',
        'note': "Discrete through-hole package screwed to an extruded heat sink with an insulating gap pad; the h "
                "of the heat sink includes the fin area (forced air). Case = back side of the Cu leadframe.",
        'chip': {'material': 'Si', 't': 0.20, 'A': 25.0},
        'inner': [L('Die attach', 'attach', 'High-Pb solder', 0.05), L('Cu leadframe', 'package', 'Cu', 2.0, 15, 20)],
        'outer': [L('Insulating pad', 'iso', 'Gap pad', 0.25), *_hs],
        'h': 2000.0, 'extra': {'R_jtop': 28.0, 'A_top': 300.0, 'h_top': 10.0, 'R_jpins': 10.0, 'R_pins_amb': 30.0}},
    'D²PAK (TO-263) on FR4 with vias': {
        'kind': 'smd',
        'note': "SMD package soldered on a thermal via array; the board is pressed with a gap pad on the housing "
                "wall (h of the housing outside: natural/forced convection or a cold plate). Case = Cu tab.",
        'chip': {'material': 'Si', 't': 0.20, 'A': 16.0},
        'inner': [L('Die attach', 'attach', 'High-Pb solder', 0.05), L('Cu tab', 'package', 'Cu', 1.3, 10, 8)],
        'outer': [L('Board solder', 'attach', 'SAC305 solder', 0.07), *_pcb_vias,
                  L('Gap pad', 'tim', 'Gap pad', 1.0, 20, 20), *_box],
        'h': 1500.0, 'extra': {'R_jtop': 40.0, 'A_top': 90.0, 'h_top': 10.0, 'R_jpins': 25.0, 'R_pins_amb': 40.0}},
    'TOLL on IMS': {
        'kind': 'smd',
        'note': "Leadless SMD package on an insulated metal substrate, IMS pressed on a cold plate with grease. "
                "Case = exposed drain pad.",
        'chip': {'material': 'Si', 't': 0.15, 'A': 16.0},
        'inner': [L('Die attach', 'attach', 'SAC305 solder', 0.04), L('Cu leadframe', 'package', 'Cu', 0.5, 10, 10)],
        'outer': [L('Board solder', 'attach', 'SAC305 solder', 0.07), L('Cu foil', 'board', 'Cu', 0.07, 14, 14),
                  L('IMS dielectric', 'board', 'IMS dielectric', 0.1), L('Al base', 'board', 'Al alloy (6063)', 1.5,
                                                                         30, 30),
                  L('Grease', 'tim', 'Thermal grease', 0.05), L('Cold plate', 'cooler', 'Al die-cast (A380)', 3.0, 60,
                                                                 60)],
        'h': 3000.0, 'extra': {'R_jtop': 30.0, 'A_top': 110.0, 'h_top': 10.0, 'R_jpins': 30.0, 'R_pins_amb': 40.0}},
    'QDPAK / TOLT top-side cooled': {
        'kind': 'topcool',
        'note': "Top-side cooled SMD package: the drain tab faces up and is cooled through an insulating pad by the "
                "heat sink; the board only carries the leads (extra path 'pins'). Case = top-side tab.",
        'chip': {'material': 'Si', 't': 0.15, 'A': 25.0},
        'inner': [L('Die attach', 'attach', 'SAC305 solder', 0.05), L('Cu tab (top)', 'package', 'Cu', 0.8, 14, 17)],
        'outer': [L('Insulating pad', 'iso', 'Gap pad', 0.3), L('Heat sink base', 'cooler', 'Al alloy (6063)', 4.0, 60,
                                                                60)],
        'h': 3000.0, 'extra': {'R_jtop': 35.0, 'A_top': 240.0, 'h_top': 10.0, 'R_jpins': 15.0, 'R_pins_amb': 25.0}},
    'Power module, Al2O3 DBC with baseplate': {
        'kind': 'module',
        'note': "Standard IGBT module: die soldered on a direct bonded copper substrate, soldered on a Cu baseplate "
                "with grease on a liquid cold plate. Case = bottom of the baseplate (R_th,jc of the datasheet).",
        'chip': {'material': 'Si', 't': 0.12, 'A': 100.0},
        'inner': [L('Die attach', 'attach', 'SAC305 solder', 0.07), L('DBC Cu (top)', 'package', 'Cu', 0.3, 25, 30),
                  L('DBC ceramic', 'iso', 'Al2O3 (96 %)', 0.38, 27, 32), L('DBC Cu (bottom)', 'package', 'Cu', 0.3,
                                                                           25, 30),
                  L('System solder', 'attach', 'SAC305 solder', 0.1), L('Cu baseplate', 'package', 'Cu', 3.0, 60, 60)],
        'outer': [L('Grease', 'tim', 'Thermal grease', 0.05), L('Cold plate', 'cooler', 'Al die-cast (A380)', 3.0, 80,
                                                                80)],
        'h': 5000.0, 'extra': {}},
    'Power module, Si3N4 AMB pin-fin, sintered (SiC)': {
        'kind': 'pinfin',
        'note': "Automotive SiC traction module: sintered die on an active metal brazed Si3N4 substrate, sintered on "
                "a Cu pin-fin baseplate that is cooled directly by the coolant (no grease); h includes the pin-fin area "
                "enhancement. Case = bottom of the substrate; datasheets often give R_th,jf (junction → fluid), "
                "which includes the baseplate.",
        'chip': {'material': 'SiC (4H)', 't': 0.18, 'A': 25.0},
        'inner': [L('Die attach (sinter)', 'attach', 'Sintered Ag', 0.03), L('AMB Cu (top)', 'package', 'Cu', 0.3, 20,
                                                                              25),
                  L('AMB ceramic', 'iso', 'Si3N4', 0.32, 22, 27), L('AMB Cu (bottom)', 'package', 'Cu', 0.3, 20, 25),
                  L('Substrate attach (sinter)', 'attach', 'Sintered Ag', 0.05)],
        'outer': [L('Cu pin-fin baseplate', 'cooler', 'Cu', 3.0, 50, 50)], 'h': 15000.0, 'extra': {}},
    'Double-side cooled module (SiC)': {
        'kind': 'dsc',
        'note': "Double-side cooled module: the die is sintered between two AMB substrates (Cu spacer on top), both "
                "sides are pressed with grease on cold plates. Case = outer Cu of each substrate.",
        'chip': {'material': 'SiC (4H)', 't': 0.18, 'A': 25.0},
        'inner': [L('Die attach (sinter)', 'attach', 'Sintered Ag', 0.03), L('AMB Cu', 'package', 'Cu', 0.3, 15, 20),
                  L('AMB ceramic', 'iso', 'Si3N4', 0.32, 17, 22), L('AMB Cu', 'package', 'Cu', 0.3, 15, 20)],
        'outer': [L('Grease', 'tim', 'Thermal grease', 0.05), L('Cold plate', 'cooler', 'Al die-cast (A380)', 2.0, 40,
                                                                40)],
        'h': 12000.0, 'extra': {},
        'second': {
            'inner': [L('Top metal', 'package', 'Al', 0.005, 5, 5), L('Spacer attach (sinter)', 'attach',
                                                                       'Sintered Ag', 0.03),
                      L('Cu spacer', 'package', 'Cu', 1.0, 5, 5), L('Spacer solder', 'attach', 'SAC305 solder', 0.1),
                      L('AMB Cu', 'package', 'Cu', 0.3, 15, 20), L('AMB ceramic', 'iso', 'Si3N4', 0.32, 17, 22),
                      L('AMB Cu', 'package', 'Cu', 0.3, 15, 20)],
            'outer': [L('Grease', 'tim', 'Thermal grease', 0.05), L('Cold plate', 'cooler', 'Al die-cast (A380)', 2.0,
                                                                    40, 40)],
            'h': 12000.0}},
    'GaN PQFN on FR4 with vias (bottom-side cooled)': {
        'kind': 'qfn',
        'note': "GaN-on-Si HEMT in a QFN package: the heat flows through the Si substrate and the exposed pad into a "
                "via array, the board is pressed with a gap pad on the housing. Case = exposed pad.",
        'chip': {'material': 'Si', 't': 0.20, 'A': 6.0},
        'inner': [L('Die attach', 'attach', 'Sintered Ag', 0.025), L('Exposed pad', 'package', 'Cu', 0.2, 6, 7)],
        'outer': [L('Board solder', 'attach', 'SAC305 solder', 0.06), *_pcb_vias,
                  L('Gap pad', 'tim', 'Gap pad', 1.0, 20, 20), *_box],
        'h': 1500.0, 'extra': {'R_jtop': 40.0, 'A_top': 64.0, 'h_top': 10.0, 'R_jpins': 30.0, 'R_pins_amb': 40.0}},
    'Embedded die in PCB (Cu microvias)': {
        'kind': 'embedded',
        'note': "Die embedded in the core of the board and connected with laser-drilled Cu microvias to a thick Cu "
                "layer that is pressed on a cold plate. Case = thick Cu layer.",
        'chip': {'material': 'Si', 't': 0.10, 'A': 16.0},
        'inner': [L('Cu microvia field', 'package', 'Cu', 0.05, 4, 4, k=round(0.3 * 390 + 0.7 * 0.3, 1)),
                  L('Cu layer (thick)', 'package', 'Cu', 0.2, 15, 15)],
        'outer': [L('Prepreg', 'board', 'FR4 (through-plane)', 0.07, 15, 15), L('Cu layer (bottom)', 'board', 'Cu',
                                                                                0.07, 30, 30),
                  L('Grease', 'tim', 'Thermal grease', 0.05), L('Cold plate', 'cooler', 'Al die-cast (A380)', 3.0, 60,
                                                                 60)],
        'h': 3000.0, 'extra': {'R_jtop': 25.0, 'A_top': 225.0, 'h_top': 10.0}},
}
PACKAGE_REFS = ['Wintrich2015', 'JESD51-14', 'Lee1995']


def preset_spec(name, dsc=None):
    """
    Side specifications of a package preset for thermStack.stackModel (inside modelled by its layers): returns
    (main, second or None). dsc: None = second side only for double-side cooled presets, True / False = force.
    """
    pk = PACKAGES[name]
    main = {'inner': {'mode': 'layers', 'chip': dict(pk['chip']), 'layers': [dict(x) for x in pk['inner']]},
            'outer': [dict(x) for x in pk['outer']], 'h': pk['h']}
    use = ('second' in pk) if dsc is None else dsc
    if not use:
        return main, None
    sec = pk.get('second') or {'inner': pk['inner'], 'outer': pk['outer'], 'h': pk['h']}
    second = {'inner': {'mode': 'layers', 'chip': dict(pk['chip'], t=1e-3), 'layers': [dict(x) for x in
                                                                                       sec['inner']]},
              'outer': [dict(x) for x in sec['outer']], 'h': sec['h']}
    return main, second


# board technologies: replace the layers with the role 'board' of a package stack
BOARDS = {
    'Keep the board of the package preset': None,
    'FR4, no thermal vias': [L('Cu pad (top layer)', 'board', 'Cu', 0.07, 12, 12),
                             L('FR4', 'board', 'FR4 (through-plane)', 1.5, 30, 30),
                             L('Cu (bottom layer)', 'board', 'Cu', 0.07, 30, 30)],
    'FR4 with thermal vias (0.3 mm, 1 mm pitch, unfilled)': _pcb_vias,
    'FR4 with filled vias (0.3 mm, 0.6 mm pitch)': [
        L('Cu pad (top layer)', 'board', 'Cu', 0.07, 12, 12),
        L('FR4 with filled vias', 'board', 'FR4 (through-plane)', 1.5, 12, 12,
          k=round(via_k(0.6, 0.3, 25, filled=True, k_fill=3.0), 2)),
        L('Cu (bottom layer)', 'board', 'Cu', 0.07, 20, 20)],
    '4-layer FR4 (2 inner Cu planes) with vias': [
        L('Cu pad (top layer)', 'board', 'Cu', 0.07, 12, 12),
        L('FR4 with vias', 'board', 'FR4 (through-plane)', 0.4, 12, 12, k=round(via_k(1.0, 0.3, 25), 2)),
        L('Inner Cu plane', 'board', 'Cu', 0.035, 40, 40),
        L('FR4 core with vias', 'board', 'FR4 (through-plane)', 0.7, 12, 12, k=round(via_k(1.0, 0.3, 25), 2)),
        L('Inner Cu plane', 'board', 'Cu', 0.035, 40, 40),
        L('FR4 with vias', 'board', 'FR4 (through-plane)', 0.4, 12, 12, k=round(via_k(1.0, 0.3, 25), 2)),
        L('Cu (bottom layer)', 'board', 'Cu', 0.07, 30, 30)],
    'Cu inlay / coin in FR4': [L('Cu inlay', 'board', 'Cu', 1.6, 10, 10)],
    'IMS (Al base, 100 µm dielectric)': [L('Cu foil', 'board', 'Cu', 0.07, 14, 14),
                                         L('IMS dielectric', 'board', 'IMS dielectric', 0.1),
                                         L('Al base', 'board', 'Al alloy (6063)', 1.5, 30, 30)],
    'Ceramic thick-film board (AlN)': [L('Thick-film Ag/Cu', 'board', 'Cu', 0.02, 14, 14),
                                       L('AlN board', 'board', 'AlN', 0.635, 30, 30)],
}

# cooling: heat transfer coefficient h on the outer surface of the last layer [W/(m2 K)] (typical range)
COOLING = {
    'Keep the h of the package preset': None,
    'Adiabatic (side not cooled)': 0.0,
    'Natural convection (flat surface)': 10.0,
    'Natural convection with fins (effective)': 50.0,
    'Forced air with fins (effective)': 1500.0,
    'Liquid cold plate (via grease)': 5000.0,
    'Direct liquid, pin-fin (effective)': 15000.0,
    'Two-phase / jet impingement': 50000.0,
}
COOLING_RANGE = {'Natural convection (flat surface)': '5–25', 'Natural convection with fins (effective)': '20–100',
                 'Forced air with fins (effective)': '500–3000', 'Liquid cold plate (via grease)': '2000–10000',
                 'Direct liquid, pin-fin (effective)': '10000–30000', 'Two-phase / jet impingement': '20000–100000'}

#######################################################################################################################
# Semiconductor devices (generic loss parameters)
#######################################################################################################################
# type: 'IGBT' (V0 + r I, antiparallel diode) or 'MOSFET' (R_ds,on, channel conducts both directions)
# V0/r and VF0/rF at T_ref; R(T) = R25 ((T + 273.15) / 298.15)^kR; E(V, I, T) = E_ref (V/V_ref)^Kv (I/I_ref)^Ki
# (1 + TcE (T - T_ref)); E_rec of the antiparallel or body diode (0 for GaN: no reverse recovery)
DEVICES = {
    'Si IGBT module 1200 V / 200 A': dict(
        tech='Si IGBT', type='IGBT', V_rated=1200, I_rated=200, V0=0.80, r=4.7e-3, dV0=-1.5e-3, dr=1.0e-5,
        VF0=0.90, rF=3.5e-3, dVF0=-2.0e-3, drF=8.0e-6, R25=0.0, kR=0.0, VSD=0.0, t_dead=1.0e-6,
        Eon=26.4e-3, Eoff=17.7e-3, Erec=7.8e-3, V_ref=600, I_ref=200, T_ref=150, Kv=1.3, Ki=1.0, Kv_rec=0.6,
        Ki_rec=0.6, TcE=0.003, Rth_jh=0.25, src="Typical 1200 V trench-field-stop IGBT module (e.g. a 2-pack 200 A "
                                                 "module, as in PyPowerSim 2MBI200XBE120)"),
    'Si IGBT discrete 1200 V / 75 A (TO-247)': dict(
        tech='Si IGBT', type='IGBT', V_rated=1200, I_rated=75, V0=1.0, r=12.0e-3, dV0=-1.5e-3, dr=3.0e-5,
        VF0=1.1, rF=12.0e-3, dVF0=-2.0e-3, drF=3.0e-5, R25=0.0, kR=0.0, VSD=0.0, t_dead=1.0e-6,
        Eon=5.15e-3, Eoff=2.95e-3, Erec=2.82e-3, V_ref=600, I_ref=75, T_ref=175, Kv=1.3, Ki=1.0, Kv_rec=0.6,
        Ki_rec=0.6, TcE=0.003, Rth_jh=0.6, src="Typical 1200 V / 75 A discrete IGBT with diode (as in PyPowerSim "
                                                "IKQ75N120CS6)"),
    'Si MOSFET 100 V / 1.5 mΩ': dict(
        tech='Si MOSFET', type='MOSFET', V_rated=100, I_rated=300, V0=0.0, r=0.0, dV0=0.0, dr=0.0, VF0=0.0, rF=0.0,
        dVF0=0.0, drF=0.0, R25=1.5e-3, kR=2.3, VSD=0.8, t_dead=0.2e-6,
        Eon=0.25e-3, Eoff=0.35e-3, Erec=0.02e-3, V_ref=48, I_ref=100, T_ref=25, Kv=1.0, Ki=1.0, Kv_rec=1.0,
        Ki_rec=0.5, TcE=0.002, Rth_jh=0.8, src="Typical low-voltage trench MOSFET (as in PyPowerSim "
                                               "IAUT300N10S5N015: tr 15 ns, tf 48 ns, Qrr 0.22 µC)"),
    'Si superjunction MOSFET 650 V / 40 mΩ': dict(
        tech='Si MOSFET', type='MOSFET', V_rated=650, I_rated=50, V0=0.0, r=0.0, dV0=0.0, dr=0.0, VF0=0.0, rF=0.0,
        dVF0=0.0, drF=0.0, R25=40e-3, kR=2.5, VSD=0.9, t_dead=0.3e-6,
        Eon=0.30e-3, Eoff=0.05e-3, Erec=1.2e-3, V_ref=400, I_ref=20, T_ref=25, Kv=1.2, Ki=1.0, Kv_rec=1.0,
        Ki_rec=0.5, TcE=0.002, Rth_jh=0.9, src="Typical 650 V superjunction MOSFET; the body diode has a large Qrr "
                                               "(≈ 10 µC), so that hard-switched bridges are not recommended"),
    'SiC MOSFET module 1200 V / 13 mΩ': dict(
        tech='SiC MOSFET', type='MOSFET', V_rated=1200, I_rated=100, V0=0.0, r=0.0, dV0=0.0, dr=0.0, VF0=0.0,
        rF=0.0, dVF0=0.0, drF=0.0, R25=13e-3, kR=1.2, VSD=3.0, t_dead=0.2e-6,
        Eon=1.48e-3, Eoff=0.35e-3, Erec=0.34e-3, V_ref=600, I_ref=75, T_ref=25, Kv=1.2, Ki=1.0, Kv_rec=1.0,
        Ki_rec=0.5, TcE=0.001, Rth_jh=0.75, src="Typical 1200 V SiC MOSFET half bridge (as in PyPowerSim "
                                                "A2F12M12W2)"),
    'SiC MOSFET discrete 750 V / 50 mΩ': dict(
        tech='SiC MOSFET', type='MOSFET', V_rated=750, I_rated=36, V0=0.0, r=0.0, dV0=0.0, dr=0.0, VF0=0.0, rF=0.0,
        dVF0=0.0, drF=0.0, R25=50e-3, kR=1.2, VSD=3.5, t_dead=0.15e-6,
        Eon=35e-6, Eoff=16e-6, Erec=10e-6, V_ref=400, I_ref=20, T_ref=25, Kv=1.2, Ki=1.0, Kv_rec=1.0, Ki_rec=0.5,
        TcE=0.001, Rth_jh=1.2, src="Typical 750 V SiC MOSFET in a top-side cooled package (as in PyPowerSim "
                                   "AIMDQ75R050M2H)"),
    'GaN HEMT 650 V / 50 mΩ': dict(
        tech='GaN HEMT', type='MOSFET', V_rated=650, I_rated=30, V0=0.0, r=0.0, dV0=0.0, dr=0.0, VF0=0.0, rF=0.0,
        dVF0=0.0, drF=0.0, R25=50e-3, kR=2.0, VSD=2.5, t_dead=0.05e-6,
        Eon=47.5e-6, Eoff=7.5e-6, Erec=0.0, V_ref=400, I_ref=15, T_ref=25, Kv=1.3, Ki=1.0, Kv_rec=1.0, Ki_rec=0.5,
        TcE=0.001, Rth_jh=1.0, src="Typical 650 V enhancement-mode GaN HEMT (no reverse recovery, reverse conduction "
                                   "with V_SD ≈ 2–3 V in the dead time, R_ds,on ≈ 2× at 150 °C)"),
    'GaN HEMT 100 V / 3 mΩ': dict(
        tech='GaN HEMT', type='MOSFET', V_rated=100, I_rated=60, V0=0.0, r=0.0, dV0=0.0, dr=0.0, VF0=0.0, rF=0.0,
        dVF0=0.0, drF=0.0, R25=3.0e-3, kR=1.8, VSD=1.8, t_dead=0.02e-6,
        Eon=12e-6, Eoff=8e-6, Erec=0.0, V_ref=48, I_ref=30, T_ref=25, Kv=1.3, Ki=1.0, Kv_rec=1.0, Ki_rec=0.5,
        TcE=0.001, Rth_jh=1.5, src="Typical 100 V enhancement-mode GaN FET (chip-scale package, 48 V DC/DC)"),
}
DEVICE_REFS = ['Wintrich2015', 'Bierhoff2004', 'Graovac2009', 'Baliga2019', 'Lidow2019']

DEVICE_PARAMS = [  # (key, symbol, unit, explanation)
    ('V0', 'V_CE0', 'V', "IGBT threshold voltage of the on-state characteristic V_CE = V_CE0 + r_CE · I at T_ref"),
    ('r', 'r_CE', 'Ω', "IGBT differential on-resistance at T_ref"),
    ('dV0', 'dV_CE0/dT', 'V/K', "temperature coefficient of V_CE0 (negative: threshold falls with T)"),
    ('dr', 'dr_CE/dT', 'Ω/K', "temperature coefficient of r_CE (positive)"),
    ('VF0', 'V_F0', 'V', "diode threshold voltage at T_ref"),
    ('rF', 'r_F', 'Ω', "diode differential resistance at T_ref"),
    ('dVF0', 'dV_F0/dT', 'V/K', "temperature coefficient of V_F0"),
    ('drF', 'dr_F/dT', 'Ω/K', "temperature coefficient of r_F"),
    ('R25', 'R_ds,on(25 °C)', 'Ω', "MOSFET/HEMT on-resistance at 25 °C"),
    ('kR', 'k_R', '-', "R_ds,on(T) = R25 ((T + 273.15)/298.15)^k_R; Si ≈ 2.3–2.5, SiC ≈ 1–1.5, GaN ≈ 2"),
    ('VSD', 'V_SD', 'V', "reverse (body diode / third quadrant) voltage during the dead time"),
    ('t_dead', 't_dead', 's', "dead time per switching transition"),
    ('Eon', 'E_on', 'J', "turn-on energy at V_ref, I_ref, T_ref (datasheet, includes the recovery current)"),
    ('Eoff', 'E_off', 'J', "turn-off energy at V_ref, I_ref, T_ref"),
    ('Erec', 'E_rec', 'J', "reverse recovery energy of the diode at V_ref, I_ref, T_ref (0 for GaN)"),
    ('V_ref', 'V_ref', 'V', "test voltage of the switching energies"),
    ('I_ref', 'I_ref', 'A', "test current of the switching energies"),
    ('T_ref', 'T_ref', '°C', "test temperature of the switching energies and of V0/r"),
    ('Kv', 'K_v', '-', "voltage exponent of E_on/E_off (≈ 1.2–1.4)"),
    ('Ki', 'K_i', '-', "current exponent of E_on/E_off (≈ 1)"),
    ('Kv_rec', 'K_v,rec', '-', "voltage exponent of E_rec (≈ 0.6)"),
    ('Ki_rec', 'K_i,rec', '-', "current exponent of E_rec (≈ 0.6)"),
    ('TcE', 'TC_E', '1/K', "relative temperature coefficient of the switching energies"),
    ('Rth_jh', 'R_th,jh', 'K/W', "thermal resistance junction–heat sink per switch (from the stack designer)"),
]

#######################################################################################################################
# Qualification tests
#######################################################################################################################
# model: lifetime model of src/tools/lifeModels.py; test: standard stress; dur: duration of the standard [h or cycles]
# n_std: samples per lot of the standard (0 failures); h_cycle: hours per cycle; slots: devices per chamber / bench;
# c_hour: cost per bench hour (overrides the general value); eol: test runs to the end-of-life criterion (maximum
# duration from the bench-time limit); raise_: test stress that may be raised when the test hardly accelerates
TESTS = {
    'HTOL / HTRB (high temperature operating life / reverse bias)': dict(
        model='Arr', unit='h', test={'T': 150.0}, dur=1000.0, n_std=77, beta=2.0,
        std="AEC-Q101 / JESD22-A108: 1000 h at T_j,max, 77 parts × 3 lots", refs=['AECQ101', 'JESD22A108']),
    'H3TRB / THB (temperature humidity bias)': dict(
        model='Peck', unit='h', test={'T': 85.0, 'RH': 85.0}, dur=1000.0, n_std=77, beta=2.0,
        std="AEC-Q101 / JESD22-A101: 1000 h at 85 °C / 85 % RH with bias", refs=['AECQ101', 'Peck1986']),
    'HTGB (high temperature gate bias, field accelerated)': dict(
        model='Eyring', unit='h', test={'T': 150.0, 'E': 4.0}, dur=1000.0, n_std=77, beta=2.0,
        std="AEC-Q101: 1000 h at T_j,max with the maximum gate voltage (oxide field E = V_GS / t_ox)",
        refs=['AECQ101', 'JEP122']),
    'TC (temperature cycling, solder joints)': dict(
        model='NL', unit='cycles', test={'dT': 205.0, 'f': 48.0, 'Tmax': 150.0}, dur=1000.0, n_std=77, beta=4.0, h_cycle=0.5,
        std="AEC-Q101 / JESD22-A104: 1000 cycles −55/150 °C", refs=['AECQ101', 'JESD22A104', 'Norris1969']),
    'PC (power cycling, bond wires / die attach)': dict(
        model='CIPS', unit='cycles', test={'dT': 100.0, 'Tmin': 40.0, 'ton': 2.0, 'I': 10.0, 'V': 12.0,
                                           'D': 400.0}, dur=30000.0, n_std=6, beta=5.0, h_cycle=6.0 / 3600,
        slots=12, c_hour=8.0, eol=True, raise_={'stress': 'dT', 'max': 150.0, 'step': 10.0},
        std="AQG 324 PC sec: ΔT_j = 100 K, t_on < 5 s, run to the end-of-life criterion (ΔV_CE +5 %, R_th +20 %); "
            "a power cycling bench holds 6–12 devices, t_on + t_off ≈ 6 s",
        refs=['AQG324', 'Bayerer2008']),
}
