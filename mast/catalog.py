"""Legacy checkpoint schema. Bounds describe synthetic training support, not grade standards."""
ELEMENT_NAMES  = ['C','Si','Mn','P','S','Cr','Ni','Mo','V','Cu','Al','Ti','Nb','B','Fe']
PROPERTY_NAMES = ['UTS','YS','elongation','reduction_of_area','HRC','impact_energy_J']
QUENCH_MAP     = {0: 'Water', 1: 'Oil', 2: 'Air', 3: 'Polymer'}
PROP_UNITS     = {'UTS': 'MPa', 'YS': 'MPa', 'elongation': '%',
                  'reduction_of_area': '%', 'HRC': '', 'impact_energy_J': 'J'}

COMP_RANGES = {
    'C':(0.05,0.80), 'Si':(0.05,0.50), 'Mn':(0.30,2.00),
    'P':(0.001,0.04), 'S':(0.001,0.04), 'Cr':(0.00,2.00),
    'Ni':(0.00,4.00), 'Mo':(0.00,0.60), 'V':(0.00,0.30),
    'Cu':(0.00,0.50), 'Al':(0.005,0.06), 'Ti':(0.00,0.10),
    'Nb':(0.00,0.10), 'B':(0.000,0.005),
}

PROP_RANGES = {
    'UTS':(400,2000,1300), 'YS':(300,1800,1100),
    'elongation':(1,35,10), 'reduction_of_area':(3,40,15),
    'HRC':(15,65,40), 'impact_energy_J':(5,140,80),
}

STEEL_PRESETS = {
    'AISI 4340': dict(C=0.40,Si=0.25,Mn=0.75,P=0.02,S=0.02,Cr=0.80,Ni=1.80,Mo=0.25,V=0.001,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'AISI 1045': dict(C=0.45,Si=0.25,Mn=0.75,P=0.02,S=0.02,Cr=0.05,Ni=0.05,Mo=0.01,V=0.001,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'AISI 4140': dict(C=0.40,Si=0.25,Mn=0.85,P=0.02,S=0.02,Cr=0.95,Ni=0.05,Mo=0.20,V=0.001,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'H13 Tool':  dict(C=0.38,Si=1.00,Mn=0.40,P=0.02,S=0.02,Cr=5.15,Ni=0.05,Mo=1.35,V=1.00,Cu=0.05,Al=0.03,Ti=0.001,Nb=0.001,B=0.0001),
    'Custom':    None,
}
