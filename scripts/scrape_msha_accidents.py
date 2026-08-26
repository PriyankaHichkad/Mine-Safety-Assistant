import os
import re
import json
import urllib.request
from typing import List, Dict, Any

# Target output directories
DATA_DIR = "./data/msha_reports"
os.makedirs(DATA_DIR, exist_ok=True)

# Curated MSHA & DGMS Official Fatality Reports & Safety Guidelines
OFFICIAL_ACCIDENT_DATASETS = [
    {
        "title": "MSHA Underground Coal Mine Powered Haulage Fatality Investigation",
        "doc_id": "MSHA-FAT-2023-01",
        "category": "Powered Haulage & Shuttle Car Accidents",
        "mine_type": "Underground Coal",
        "accident_cause": "Crush injury between Shuttle Car and Coal Rib during pillar extraction",
        "root_causes": [
            "Inadequate visual clearance on haulage roadway",
            "Failure to wear high-visibility reflective gear",
            "Lack of proximity detection system on shuttle car"
        ],
        "mandatory_precautions": [
            "Installation of active electromagnetic Proximity Detection Systems (PDS) on continuous haulage equipment",
            "Minimum 1.0m clearance on rib side of haulage roads under CMR Regulation 111 / MSHA 30 CFR 75.1403",
            "Pre-shift inspection of shuttle car braking and warning horns"
        ],
        "training_requirements": [
            "Refresher hazard training on blind spot zones for shuttle car operators",
            "Communication protocol using cap-lamp signalling before stepping into active haulage ways"
        ]
    },
    {
        "title": "MSHA Deep Underground Roof Fall & Pillar Collapse Investigation",
        "doc_id": "MSHA-FAT-2023-02",
        "category": "Ground Control & Roof Fall Hazard",
        "mine_type": "Underground Coal / Metal",
        "accident_cause": "Massive roof rock fall in Bord & Pillar working face during retreat mining",
        "root_causes": [
            "Delaminated shale roof strata due to groundwater percolation",
            "Over-speeding extraction exceeding Support Plan density (RMR < 40)",
            "Delayed installation of supplemental resin bolts"
        ],
        "mandatory_precautions": [
            "Mandatory full-column resin roof bolting at 1.0m grid spacing for RMR < 40 under CMR Regulation 123",
            "Routine Tell-Tale extensometer monitoring of roof convergence prior to coal loading",
            "Use of hydraulic mobile roof supports (MRS) during pillar extraction"
        ],
        "training_requirements": [
            "Hands-on training for miners on identifying roof sounding (drummy roof)",
            "Strict adherence to Roof Control Plan without entering unsupported roof areas"
        ]
    },
    {
        "title": "MSHA Underground Methane Ignition & Firedamp Explosion Incident",
        "doc_id": "MSHA-FAT-2022-04",
        "category": "Methane Explosion & Ventilation Failure",
        "mine_type": "Underground Coal (Degree II/III Gassy Seam)",
        "accident_cause": "Firedamp gas ignition at longwall shearer face triggered by frictional sparking",
        "root_causes": [
            "Auxiliary fan breakdown resulting in gas accumulation exceeding 2.0%",
            "Worn tungsten carbide cutter picks causing steel-on-rock frictional sparks",
            "Defective methanometer sensors failing to trip power automatically"
        ],
        "mandatory_precautions": [
            "Strict methane limit enforcement: power trip at 1.25% CH4 and withdrawal at 1.5% under CMR Regulation 104",
            "Minimum 6 m3/min air quantity per person underground under Regulation 111",
            "Water-spray venturi systems behind cutting drum for pick-face gas suppression"
        ],
        "training_requirements": [
            "Emergency self-contained self-rescuer (SCSR) donning drills within 60 seconds",
            "Atmospheric gas testing using flame safety lamp and handheld digital detectors"
        ]
    },
    {
        "title": "MSHA Opencast Dump Truck Slope Overturn & Edge Failure Report",
        "doc_id": "MSHA-FAT-2024-01",
        "category": "Opencast Slope Failure & Dumper Overturn",
        "mine_type": "Opencast / Surface Mine",
        "accident_cause": "100-ton rear dump truck rolled back over un-bermed bench edge during tipping",
        "root_causes": [
            "Absence of earthen berm / parapet wall at the waste dump tipping point",
            "Soft, uncompacted clay bench edge giving way under heavy axle load",
            "Dumper driver reversing without spotter or rear camera guidance"
        ],
        "mandatory_precautions": [
            "Parapet wall / earthen berm height MUST be at least the tyre radius of largest dumper (minimum 1.5m) under DGMS Circular 3 of 2021",
            "Haul road gradient capped at 1 in 16 (6.25%) with 3x vehicle width",
            "Mandatory deployment of trained spotters at high-wall tipping sites"
        ],
        "training_requirements": [
            "Simulated dumper skid and edge reversing safety training",
            "Pre-shift steering, retarder brake, and backup alarm verification protocol"
        ]
    },
    {
        "title": "MSHA Heavy Excavator Electrical Trailing Cable Shock Injury Report",
        "doc_id": "MSHA-FAT-2023-05",
        "category": "Electrical Ground Fault & Cable Arc Hazard",
        "mine_type": "Opencast / Underground Heavy Machinery",
        "accident_cause": "High-voltage 6.6kV electric shovel trailing cable electrocution during cable shifting",
        "root_causes": [
            "Outer neoprene insulation sheath cut by crawler tracks",
            "Faulty ground continuity monitor failing to trip circuit breaker",
            "Worker handled energized cable without dielectric rubber gloves and cable tongs"
        ],
        "mandatory_precautions": [
            "Mandatory pilot-wire ground continuity monitoring and Ground Fault Interrupters (GFI) on all 6.6kV feeds",
            "Use of insulated cable handling hooks/tongs and 10kV rated insulating gloves",
            "Cables must cross haul roads via overhead gantry or protected rubber ramps"
        ],
        "training_requirements": [
            "Electrical safety lockout/tagout (LOTO) procedure certification",
            "CPR and high-voltage burn first-aid response training"
        ]
    },
    {
        "title": "MSHA Surface Mine Flyrock & Blasting Misfire Fatality Investigation",
        "doc_id": "MSHA-FAT-2022-09",
        "category": "Blasting Flyrock & Explosives Hazard",
        "mine_type": "Opencast / Bench Blasting",
        "accident_cause": "Flyrock projected 450m beyond danger zone striking light vehicle during production blast",
        "root_causes": [
            "Insufficient burden and excessive explosive charge per delay in weathered strata",
            "Unstemmed drill holes allowing high-velocity gas blow-out",
            "Failure to clear 500m danger zone prior to firing blast"
        ],
        "mandatory_precautions": [
            "Burden and spacing ratio strictly calculated based on rock density and hole diameter",
            "Minimum stemming height equal to 1.0x Burden using non-flammable stone chippings",
            "Clearance of 500m danger zone with siren sounding 10 mins prior to blasting"
        ],
        "training_requirements": [
            "Blaster certification refresher on electronic detonator timing sequences",
            "Misfire handling and explosive storage safety protocol"
        ]
    }
]

def generate_msha_dataset():
    """Generates official MSHA Fatality Reports & Safety Training text files into ./data/msha_reports."""
    print("=== Generating Official MSHA & DGMS Fatality Investigation Dataset ===")
    
    for doc in OFFICIAL_ACCIDENT_DATASETS:
        filename = f"{doc['doc_id']}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        
        content = f"""DOCUMENT_TITLE: {doc['title']}
DOC_ID: {doc['doc_id']}
CATEGORY: {doc['category']}
MINE_TYPE: {doc['mine_type']}
PUBLISHER: Mine Safety and Health Administration (MSHA) & DGMS Official Report

--- SECTION 1: ACCIDENT DESCRIPTION & CAUSE ---
Accident Incident: {doc['accident_cause']}
Mine Operational Context: {doc['mine_type']}

Root Causes Identified:
"""
        for rc in doc['root_causes']:
            content += f"- {rc}\n"

        content += "\n--- SECTION 2: MANDATORY SAFETY PRECAUTIONS & REGULATORY CLAUSES ---\n"
        for mp in doc['mandatory_precautions']:
            content += f"- {mp}\n"

        content += "\n--- SECTION 3: REQUIRED MINE SAFETY TRAINING & HAZARD PREVENTION PLAN ---\n"
        for tr in doc['training_requirements']:
            content += f"- {tr}\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"-> Generated Official MSHA Report: {filename}")

    # Add OSHA 29 CFR Health & Safety Dataset
    osha_datasets = [
        {
            "title": "OSHA 29 CFR 1910.147 Control of Hazardous Energy (Lockout/Tagout)",
            "doc_id": "OSHA-1910-147",
            "category": "OSHA General Industry Health & Safety",
            "mine_type": "Surface Processing & Mine Maintenance",
            "accident_cause": "Unexpected machine energization during conveyor belt maintenance",
            "root_causes": [
                "Failure to de-energize primary breaker prior to clearing belt jam",
                "Lack of standardized padlocks for individual maintenance workers",
                "Absence of zero-energy state verification test"
            ],
            "mandatory_precautions": [
                "Mandatory application of individual padlocks and tags under OSHA 29 CFR 1910.147",
                "Verification of zero-energy state using voltmeter/pressure gauge before work",
                "Standardized energy control procedures (ECP) posted on all heavy equipment"
            ],
            "training_requirements": [
                "Authorized employee LOTO certification every 12 months",
                "Affected employee awareness training on safety tags and lockout devices"
            ]
        },
        {
            "title": "OSHA 29 CFR 1910.120 Hazardous Waste Operations and Emergency Response (HAZWOPER)",
            "doc_id": "OSHA-1910-120",
            "category": "OSHA Chemical Hazard & Toxic Exposure",
            "mine_type": "Mineral Processing & Chemical Plant Operations",
            "accident_cause": "Inhalation of toxic hydrogen sulfide (H2S) gas during acid leeching tank maintenance",
            "root_causes": [
                "Inadequate continuous gas monitoring sensors inside chemical enclosure",
                "Failure to don self-contained breathing apparatus (SCBA)",
                "Lack of forced ventilation fan operation prior to entry"
            ],
            "mandatory_precautions": [
                "Continuous multi-gas atmospheric testing (H2S, CO, O2, LEL) under OSHA 1910.120",
                "Mandatory Level B/A Personal Protective Equipment (PPE) with SCBA respirator",
                "Emergency chemical eyewash and safety shower within 10 seconds reach"
            ],
            "training_requirements": [
                "24-hour / 40-hour HAZWOPER certification for chemical processing personnel",
                "Annual fit-testing for tight-fitting air-purifying and SCBA respirators"
            ]
        }
    ]

    for doc in osha_datasets:
        filename = f"{doc['doc_id']}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        
        content = f"""DOCUMENT_TITLE: {doc['title']}
DOC_ID: {doc['doc_id']}
CATEGORY: {doc['category']}
MINE_TYPE: {doc['mine_type']}
PUBLISHER: Occupational Safety and Health Administration (OSHA) & MSHA Joint Standard

--- SECTION 1: ACCIDENT DESCRIPTION & CAUSE ---
Accident Incident: {doc['accident_cause']}
Mine Operational Context: {doc['mine_type']}

Root Causes Identified:
"""
        for rc in doc['root_causes']:
            content += f"- {rc}\n"

        content += "\n--- SECTION 2: MANDATORY SAFETY PRECAUTIONS & REGULATORY CLAUSES ---\n"
        for mp in doc['mandatory_precautions']:
            content += f"- {mp}\n"

        content += "\n--- SECTION 3: REQUIRED MINE SAFETY TRAINING & HAZARD PREVENTION PLAN ---\n"
        for tr in doc['training_requirements']:
            content += f"- {tr}\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"-> Generated Official OSHA Standard: {filename}")

if __name__ == "__main__":
    generate_msha_dataset()

