# VASP-Python
This is the data file associated with the publication "User-defined Electrostatic Potentials in DFT Supercell Calculations: Implementation and Application to Electrified Interfaces"

`H_in_constant_field` is the example of one H atom in a constant field (Figure 2b).

`CCE` is the Ne computational counter electrode (Figure 3a).

`CDCE` is the charge density counter electrode (Figure 3b).

# Parameters
At the beginning of each `vasp_plugin.py` file, there are calculation specific parameters that need to be adapted for each system. The units are eV for energy, eV/Å for force, Å for length, $e$ for charge.  

`Q_pos` is the position of the counter electrode in Cartesian coordinate. In the examples, the conter electrode is a Gaussian wall which is constant in the x and y direction, so the x and y coordinates here do not change the setup.

`Q0` is the charge of the counter electrode at step 0.

`nelect_neutral` is the total number of electrons in the system when the cell is charge-neutral. This number is used for checking the charge normalization.

`grid_roll_frac` is only used to compute the dipole correction in the python plugin. When integrating the charge density to get the dipole moment, the charge density is rolled in the z direction by `grid_roll_frac` (in fractional coordinate). This is to ensure that the electron charge density and the core charge density of an given atom stay in the same periodic image. In other words, after the rolling, the charge density at z = 0 should be 0.   

`grid_position_frac` is the z position of the dipole correction (fractional coordinate).

## CCE-specific parameters
`width_wall` is the width of the wall for confining water in CDCE. 

`pos_right_wall` is the z position of the wall for confining water in CDCE (Cartesian coordinate).

## CCE-specific parameters
`i_Ne` is the index of the Ne element following the order in POSCAR. The starting index is 0. (For example, if the element order in POSCAR is "Au Ne H O", `i_Ne` should be 1.)

`n_elements` is the total number of element. (If the element order in POSCAR is "Au Ne H O", `n_elements` should be 4.)

`n_Ne` is the total number of Ne atoms. 

## thermopotentiostat-specific parameters
`phi0` is the target electrode potential measured at the dipole correction (see the discussion of the thermopotentiostat in the paper)

`tau` and `temperature` determines the fluctuation and disspation of the thermopotentiostat. (See the discussions in paper, Eq. 6.)

`C0` is the bare capacitance of the electrodes in vacuum, which is calculated by `C0 = eps0*A/d_electrode`. Here `eps0` is the vacuum permitivity, `A` the surface area, and `d_electrode` the distance between the two electrode.    

