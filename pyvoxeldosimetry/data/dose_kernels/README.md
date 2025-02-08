# Dose Kernel Data Structure

Each radionuclide folder contains the following files:
- `kernel_1mm.dat`: Dose point kernel data at 1mm resolution
- `kernel_2mm.dat`: Dose point kernel data at 2mm resolution
- `kernel_3mm.dat`: Dose point kernel data at 3mm resolution
- `metadata.json`: Contains kernel properties and reference information

Supported radionuclides:
1. F-18 (positron emitter)
2. Ga-68 (positron emitter)
3. Cu-64 (positron emitter/electron capture)
4. Zr-89 (positron emitter)
5. Y-90 (beta emitter)
6. I-131 (beta/gamma emitter)
7. Lu-177 (beta/gamma emitter)
8. Tb-161 (beta emitter)
9. Ac-225 (alpha emitter)
10. Pb-212 (alpha/beta emitter)

Each kernel file contains:
- Energy deposition data
- Range information
- Radiation type specific parameters