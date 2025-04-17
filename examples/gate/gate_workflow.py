import os
import subprocess
import shutil

import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ---- CONFIGURATION ----
GATE_DOCKER_IMAGE = "opengatecollaboration/gate:9.4-docker"
GATE_RESOURCES_DIR = Path(__file__).parent / "data"
# ---- ISOTOPE MAPPING TABLE ----
ISOTOPE_TABLE = {
    "Y90":   {"Z": 39, "A": 90,  "charge": 0},
    "Lu177": {"Z": 71, "A": 177, "charge": 0},
    "I131":  {"Z": 53, "A": 131, "charge": 0},
    "Tb161": {"Z": 65, "A": 161, "charge": 0},
    "Ac225": {"Z": 89, "A": 225, "charge": 0},
    "Pb212": {"Z": 82, "A": 212, "charge": 0},
    "Ga68":  {"Z": 31, "A": 68,  "charge": 0},
    "F18":   {"Z": 9,  "A": 18,  "charge": 0},
    "Tc99m": {"Z": 43, "A": 99,  "charge": 0},
    "Zr89":  {"Z": 40, "A": 89,  "charge": 0},
    # Add more isotopes as needed
}

def read_mhd_image(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return arr, spacing

def write_mhd_image(arr, spacing, out_path):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(out_path))

def prepare_gate_inputs(activity_map_path, ct_image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    act_arr, act_spacing = read_mhd_image(activity_map_path)
    ct_img = sitk.ReadImage(str(ct_image_path))
    # Resample CT to match activity map
    act_img = sitk.ReadImage(str(activity_map_path))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(act_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    ct_img_resampled = resampler.Execute(ct_img)
    ct_arr = sitk.GetArrayFromImage(ct_img_resampled)
    ct_spacing = ct_img_resampled.GetSpacing()
    write_mhd_image(act_arr, act_spacing, output_dir / "ACTM.mhd")
    write_mhd_image(ct_arr, ct_spacing, output_dir / "CTRS.mhd")
    # Copy GATE resource files
    shutil.copy(GATE_RESOURCES_DIR / "GateMaterials.db", output_dir / "GateMaterials.db")
    shutil.copy(GATE_RESOURCES_DIR / "MaterialsTable.txt", output_dir / "MaterialsTable.txt")
    shutil.copy(GATE_RESOURCES_DIR / "DensitiesTable.txt", output_dir / "DensitiesTable.txt")
    return act_arr, act_spacing

def generate_gate_macros(output_dir, isotope, total_activity, act_arr, act_spacing):
    mac_dir = Path(output_dir) / "mac"
    mac_dir.mkdir(exist_ok=True)
    # Calculate FOV and world size
    res_z, res_x, res_y = act_arr.shape
    vox_x, vox_y, vox_z = act_spacing
    fov_x = res_x * vox_x
    fov_y = res_y * vox_y
    fov_z = res_z * vox_z
    world_x = fov_x + 200  # mm
    world_y = fov_y + 200  # mm
    world_z = fov_z + 200  # mm

    # --- Isotope lookup ---
    iso = ISOTOPE_TABLE.get(isotope)
    if iso is None:
        raise ValueError(f"Isotope '{isotope}' not found in ISOTOPE_TABLE.")

    # Geometry macro
    with open(mac_dir / "geometry.mac", "w") as f:
        f.write(f"""
/gate/geometry/setMaterialDatabase /APP/data/GateMaterials.db
/gate/world/geometry/setXLength {world_x/10:.1f} cm
/gate/world/geometry/setYLength {world_y/10:.1f} cm
/gate/world/geometry/setZLength {world_z/10:.1f} cm
/gate/HounsfieldMaterialGenerator/SetMaterialTable /APP/data/MaterialsTable.txt
/gate/HounsfieldMaterialGenerator/SetDensityTable /APP/data/DensitiesTable.txt
/gate/HounsfieldMaterialGenerator/SetDensityTolerance 0.005 g/cm3
/gate/HounsfieldMaterialGenerator/SetOutputMaterialDatabaseFilename /APP/data/HUmaterials.db
/gate/HounsfieldMaterialGenerator/SetOutputHUMaterialFilename /APP/data/HU2Mat.txt
/gate/HounsfieldMaterialGenerator/Generate
/gate/world/daughters/name GlobalBox
/gate/world/daughters/insert ImageNestedParametrisedVolume
/gate/geometry/setMaterialDatabase /APP/data/HUmaterials.db
/gate/GlobalBox/geometry/setHUToMaterialFile /APP/data/HU2Mat.txt
/gate/GlobalBox/geometry/setImage /APP/data/CTRS.mhd
/gate/GlobalBox/vis/forceWireframe
""")
    # Actor macro
    res_z, res_x, res_y = act_arr.shape
    vox_x, vox_y, vox_z = act_spacing
    with open(mac_dir / "actor.mac", "w") as f:
        f.write(f"""
/gate/actor/addActor DoseActor doseDistribution
/gate/actor/doseDistribution/attachTo GlobalBox
/gate/actor/doseDistribution/stepHitType random
/gate/actor/doseDistribution/setPosition 0 0 0 mm
/gate/actor/doseDistribution/setResolution {res_x} {res_y} {res_z}
/gate/actor/doseDistribution/setVoxelSize {vox_x} {vox_y} {vox_z} mm
/gate/actor/doseDistribution/setDoseAlgorithm MassWeighting
/gate/actor/doseDistribution/enableEdep true
/gate/actor/doseDistribution/enableUncertaintyEdep true
/gate/actor/doseDistribution/enableDose true
/gate/actor/doseDistribution/enableUncertaintyDose true
/gate/actor/doseDistribution/save /APP/output/doseMonteCarlo.mhd
/gate/actor/doseDistribution/saveEdep /APP/output/edep.mhd
/gate/actor/doseDistribution/saveUncertaintyEdep /APP/output/edep_uncertainty.mhd
/gate/actor/doseDistribution/saveUncertaintyDose /APP/output/dose_uncertainty.mhd
/gate/actor/doseDistribution/exportMassImage /APP/output/doseMonteCarlo-Mass.mhd
/gate/actor/addActor SimulationStatisticActor stat
/gate/actor/stat/save /APP/output/SimulationStatus.txt
""")
    # Physics macro
    with open(mac_dir / "physics.mac", "w") as f:
        f.write("""
/gate/physics/addPhysicsList emstandard_opt4
/gate/physics/addProcess RadioactiveDecay GenericIon
/gate/physics/addProcess Decay
/gate/physics/addProcess alpha
/gate/physics/addProcess betaMinus
/gate/physics/addProcess betaPlus
/gate/physics/addProcess gamma
/gate/physics/enableAlpha true
/gate/physics/enableBetaMinus true
/gate/physics/enableBetaPlus true
/gate/physics/enableGamma true
/gate/physics/enableDaughterDecay true
/gate/physics/Gamma/SetCutInRegion world 1.0 mm
/gate/physics/Electron/SetCutInRegion world 1.0 mm
/gate/physics/Alpha/SetCutInRegion world 1.0 mm
/gate/physics/Positron/SetCutInRegion world 1.0 mm
/gate/physics/Gamma/SetCutInRegion GlobalBox 0.1 mm
/gate/physics/Electron/SetCutInRegion GlobalBox 0.1 mm
/gate/physics/Alpha/SetCutInRegion GlobalBox 0.1 mm
/gate/physics/Positron/SetCutInRegion GlobalBox 0.1 mm
/gate/physics/setEMin 0.1 keV
/gate/physics/setEMax 10 GeV
/gate/physics/setDEDXBinning 220
/gate/physics/setLambdaBinning 220
""")
    # Source macro
    with open(mac_dir / "source.mac", "w") as f:
        f.write(f"""
/gate/source/addSource {isotope} voxel
/gate/source/{isotope}/reader/insert image
/gate/source/{isotope}/imageReader/translator/insert linear
/gate/source/{isotope}/imageReader/linearTranslator/setScale 1 Bq
/gate/source/{isotope}/imageReader/readFile /APP/data/ACTM.mhd
/gate/source/{isotope}/setPosition 0 0 0 mm
/gate/source/{isotope}/gps/particle ion
/gate/source/{isotope}/gps/ion {iso['Z']} {iso['A']} 0 {iso['charge']}
/gate/source/{isotope}/gps/ene/mono 0. keV
/gate/source/{isotope}/setForcedUnstableFlag true
/gate/source/{isotope}/gps/ang/type iso
/gate/source/{isotope}/gps/ene/type Mono
/gate/source/list
""")
    # Executor macro
    with open(mac_dir / "executor.mac", "w") as f:
        f.write(f"""
/control/execute mac/geometry.mac
/control/execute mac/physics.mac
/control/execute mac/actor.mac
/gate/run/initialize
/control/execute mac/source.mac
/gate/application/setTotalNumberOfPrimaries {int(total_activity)}
/gate/application/start
exit
""")
    return mac_dir / "executor.mac"

def run_gate_simulation(gate_folder, macro_file):
    cmd = [
        "docker", "run", "-ti", "--rm",
        "-v", f"{gate_folder}:/APP",
        GATE_DOCKER_IMAGE,
        f"/APP/mac/{macro_file.name}"
    ]
    print("Running GATE:", " ".join(cmd))
    subprocess.run(cmd)

def collect_and_calculate_doserate(output_dir):
    # Read GATE output dose image
    dose_path = Path(output_dir) / "output" / "doseMonteCarlo.mhd"
    dose_arr, spacing = read_mhd_image(dose_path)
    # If number of primaries = total activity (Bq), dose_arr is already Gy/s
    # Save doserate as MHD (Gy/s)
    doserate_path = Path(output_dir) / "output" / "doserate.mhd"
    write_mhd_image(dose_arr, spacing, doserate_path)
    print(f"Doserate map saved to {doserate_path}")
    return doserate_path

if __name__ == "__main__":
    # Example usage
    ACTIVITY_MAP_PATH = "/mnt/d/DATA/010_Dosimetry/001_Gate/Case-020/Inputs/Y90_PET.mhd"
    CT_IMAGE_PATH = "/mnt/d/DATA/010_Dosimetry/001_Gate/Case-020/Inputs/CT.mhd"
    ISOTOPE = "Y90"
    OUTPUT_DIR = "/mnt/d/WSL/workspace/devhliu/dosimetry/PyVoxelDosimetry/examples/gatesim"
    CALIBRATION = 1.0  # If needed, scale activity map

    act_arr, act_spacing = prepare_gate_inputs(ACTIVITY_MAP_PATH, CT_IMAGE_PATH, f"{OUTPUT_DIR}/data")
    total_activity = np.sum(act_arr) * CALIBRATION
    executor_mac = generate_gate_macros(OUTPUT_DIR, ISOTOPE, total_activity, act_arr, act_spacing)
    # Uncomment to run GATE (requires Docker and GATE image)
    run_gate_simulation(OUTPUT_DIR, executor_mac)
    # Uncomment to process results after GATE run
    # collect_and_calculate_doserate(OUTPUT_DIR)