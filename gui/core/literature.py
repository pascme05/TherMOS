#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         literature.py
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Literature of the Theory & Tools pages (subset of the OPTELIA literature list used by the thermal stack and the
loss calculator): REFS {key: {type, label, cite, url}}, url(key), label(key).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import urllib.parse

#######################################################################################################################
# References
#######################################################################################################################
REFS = {'Incropera2007': {'type': 'Book', 'label': 'Incropera et al. 2007', 'cite': 'F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, Fundamentals of Heat and Mass Transfer, 6th ed., John Wiley & Sons, 2007.', 'url': None, 'topic': 'Heat conduction, convection and material properties', 'modes': [0]},
        'Wintrich2015': {'type': 'Book', 'label': 'Wintrich et al. 2015', 'cite': 'A. Wintrich, U. Nicolai, W. Tursky and T. Reimann, Application Manual Power Semiconductors, 2nd ed., SEMIKRON International, 2015.', 'url': None, 'topic': 'Losses, thermal Foster/Cauer networks and lifetime of power semiconductors', 'modes': [0]},
        'JESD51-14': {'type': 'Norm / standard / guideline', 'label': 'JEDEC JESD51-14', 'cite': 'JEDEC JESD51-14, Transient Dual Interface Test Method for the Measurement of the Thermal Resistance Junction-to-Case of Semiconductor Devices with Heat Flow Through a Single Path, JEDEC, 2010.', 'url': 'https://www.jedec.org', 'topic': 'Thermal impedance and Foster networks of semiconductors', 'modes': [0]},
        'Lee1995': {'type': 'Paper', 'label': 'Lee et al. 1995', 'cite': 'S. Lee, S. Song, V. Au and K. P. Moran, "Constriction/spreading resistance model for electronics packaging," Proc. ASME/JSME Thermal Engineering Joint Conference, vol. 4, pp. 199–206, 1995.', 'url': None, 'topic': 'Spreading resistance of a heated plate with convection (thermal stack designer)', 'modes': [0]},
        'Bierhoff2004': {'type': 'Paper', 'label': 'Bierhoff & Fuchs 2004', 'cite': 'F. Bierhoff and F. W. Fuchs, "Semiconductor losses in voltage source and current source IGBT converters based on analytical derivation," Proc. IEEE 35th Power Electronics Specialists Conference (PESC), pp. 2836–2842, 2004.', 'url': None, 'topic': 'Closed-form inverter losses with sinusoidal PWM', 'modes': [0]},
        'Graovac2009': {'type': 'Paper', 'label': 'Graovac & Pürschel 2009', 'cite': 'D. Graovac and M. Pürschel, "IGBT power losses calculation using the data-sheet parameters," Application Note, Infineon Technologies, 2009.', 'url': None, 'topic': 'Conduction and switching losses of IGBTs and diodes in inverters', 'modes': [0]},
        'Baliga2019': {'type': 'Book', 'label': 'Baliga 2019', 'cite': 'B. J. Baliga, Fundamentals of Power Semiconductor Devices, 2nd ed., Springer, 2019.', 'url': 'https://doi.org/10.1007/978-3-319-93988-9', 'topic': 'Si and SiC power devices, on-resistance and its temperature dependence', 'modes': [0]},
        'Lidow2019': {'type': 'Book', 'label': 'Lidow et al. 2019', 'cite': 'A. Lidow, M. de Rooij, J. Strydom, D. Reusch and J. Glaser, GaN Transistors for Efficient Power Conversion, 3rd ed., John Wiley & Sons, 2019.', 'url': None, 'topic': 'GaN HEMT characteristics, reverse conduction and switching losses', 'modes': [0]},
        'Dowell1966': {'type': 'Journal paper', 'label': 'Dowell 1966', 'cite': 'P. L. Dowell, Effects of eddy currents in transformer windings, Proceedings of the IEE, vol. 113, no. 8, pp. 1387-1394, 1966.', 'url': 'https://doi.org/10.1049/piee.1966.0236', 'topic': 'Skin and proximity losses of windings (layers of foil or round wire)', 'modes': [0]},
        'Kazimierczuk2014': {'type': 'Book', 'label': 'Kazimierczuk 2014', 'cite': 'M. K. Kazimierczuk, High-Frequency Magnetic Components, 2nd ed., John Wiley & Sons, 2014.', 'url': None, 'topic': 'Skin depth, AC resistance of round wires (Bessel solution), proximity effect', 'modes': [0]},
        'AQG324': {'type': 'Norm / standard / guideline', 'label': 'ECPE AQG 324', 'cite': 'ECPE AQG 324, Qualification of Power Modules for Use in Power Electronics Converter Units in Motor Vehicles, ECPE.', 'url': 'https://www.ecpe.org', 'topic': 'Power cycling and qualification of power modules', 'modes': [0, 1, 5]}}


#######################################################################################################################
# Functions
#######################################################################################################################
def url(key):
    """Link of a reference: DOI or issuer if known, else a Google Scholar search."""
    r = REFS[key]
    if r.get('url'):
        return r['url']
    return 'https://scholar.google.com/scholar?q=' + urllib.parse.quote(r['cite'])


def label(key):
    return REFS[key]['label']
