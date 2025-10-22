import itertools
import numpy as np
import time
from scipy.special import comb
import os


Q = 0.1  # normal charge on the right 
Q_pos_1 = np.array([5.800255505, 5.023168615, 20]) # position of the first charge density
Q_pos_2 = np.array([5.800255505, 5.023168615, 0])  # position of the second charge density
rQ_list = [Q_pos_1, Q_pos_2]
Q_list = [Q, -Q]
nelect_neutral =  1  # total electron of the neutral cell
grid_roll_frac = 0.2 # z reference position of the dipole  
grid_position_frac = 0.85 # z position of the dipole correction (fractional coordinate) 

def _compute_reciprocal_lattice_vectors(cell):
    volume = _compute_volume(cell)
    b1 = 2 * np.pi * np.cross(cell[1], cell[2]) / volume
    b2 = 2 * np.pi * np.cross(cell[2], cell[0]) / volume
    b3 = 2 * np.pi * np.cross(cell[0], cell[1]) / volume
    return np.array([b1, b2, b3])

def external_charge(nx, ny, nz, q, r, box, mode='gaussianWall'):
    x = np.linspace(0, 1.0, nx,endpoint=False)
    y = np.linspace(0, 1.0, ny,endpoint=False)
    z = np.linspace(0, 1.0, nz,endpoint=False)
    Lz = box[2][2]
    xx, yy, zz = np.tensordot(box, np.meshgrid(x,y,z, indexing='ij'), axes=1)
    dV = np.abs(np.linalg.det(box)) / (nx*ny*nz)
    rho = np.zeros((nx, ny, nz))
    if(mode=='gaussianWall'):
        sigma2 = .3**2
        Area = np.linalg.norm(np.cross(box[:,0], box[:,1]))
        for i, r_i in enumerate(r):
            delta = zz-r_i[2]
            delta -= Lz * np.round(delta / Lz)
            rho += q[i]/(np.sqrt(2*np.pi*sigma2)) * np.exp(-0.5*delta**2/sigma2) / Area
    elif(mode=='deltaPeak'):
        rho[int(r[0]/ax *nx), int(r[1]/ay * ny), int(r[2]/az * nz)] = q / dV
    else:
        print('!!!Electrode-shape not supported!!!', flush=True)
        raise Exception('Electrode-shape not supported')
    check_normalization(q, np.sum(rho)*dV)
    return rho, -electrostatic_potential(rho, box[0][0], box[1][1], box[2][2], epsilon_0=0.005526349358057108)
    # rho is normal charge, electrostatic_potential gives normal pot, then the minus sign gives electron potential

def _compute_volume(cell):
    volume = np.dot(np.cross(cell[0], cell[1]), cell[2])
    return volume


def _compute_gvectors(cell, grid_size):
    g_vectors = np.zeros(list(grid_size) + [3])
    fourier_frequencies = np.zeros(list(grid_size) + [3])
    b1, b2, b3 = _compute_reciprocal_lattice_vectors(cell)
    grid_x = range(grid_size[0])
    grid_y = range(grid_size[1])
    grid_z = range(grid_size[2])
    for i, j, k in itertools.product(grid_x, grid_y, grid_z):
        ii = i
        jj = j
        kk = k
        if i > grid_size[0] // 2:
            ii -= grid_size[0]
        if j > grid_size[1] // 2:
            jj -= grid_size[1]
        if k > grid_size[2] // 2:
            kk -= grid_size[2]
        g1 = ii * b1
        g2 = jj * b2
        g3 = kk * b3
        g_vectors[i, j, k, :] = g1 + g2 + g3
        fourier_frequencies[i, j, k, :] = np.array([ii, jj, kk])
    return g_vectors, fourier_frequencies


def _safe_divide(quantity_1, quantity_2):
    safe = np.divide(
        quantity_1, quantity_2, out=np.zeros_like(quantity_1), where=(quantity_2 != 0)
    )
    return safe

def check_normalization(q, rhoSum, tol=1e-3):
    return np.abs(q-rhoSum) < tol
        
def external_potential_gradient(Lx, Ly, Lz, nx, ny, nz, dV):
    z = np.linspace(0, Lz, nz, endpoint=False)
    dz = z[1] - z[0]
    dV_gradient = np.zeros((nx, ny, nz))
    dV_gradient = -(np.roll(dV, 1, axis = 2) - np.roll(dV, -1, axis = 2))/(2*dz)
    return dV_gradient

def force_ion_from_external(
    lattice_vectors: np.ndarray,
    number_ion_types: float,
    ion_types: np.ndarray,
    number_ions: np.ndarray,
    zval: np.ndarray,
    positions: np.ndarray,
    charge_density: np.ndarray,
):
    volume = _compute_volume(lattice_vectors)
    grid_size = charge_density.shape
    nx, ny, nz = charge_density.shape
    box = lattice_vectors
    Lx = box[0][0]
    Ly = box[1][1]
    Lz = box[2][2]
    delta_pot = np.load('delta_pot.npy')
    # delta_pot = data
    # np.save('delta_pot_in_force_calc.npy', delta_pot)
    forces = np.zeros((positions.shape[0], 3)) 
    dV_gradient = external_potential_gradient(Lx, Ly, Lz, nx, ny, nz, delta_pot)
    for i in range(number_ions):
        pos = positions[i]
        pos_x = int(np.round(pos[0]*nx))%nx
        pos_y = int(np.round(pos[1]*ny))%ny
        pos_z = int(np.round(pos[2]*nz))%nz
        local_V_gradient = dV_gradient[pos_x, pos_y, pos_z]  
        forces[i, 2] += zval[ion_types[i]]*local_V_gradient
    return forces


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    return result
    
def wall(r, scaling=1):
    forces = np.zeros_like(r)    
    z = r[:,2]
    fz = np.zeros_like(z)
    if(pos_right_wall>-1):
        fz += -smoothstep(z, x_min=pos_right_wall-width_wall/2, x_max=pos_right_wall+width_wall/2, N=4)
    forces[:,2] += scaling*fz
    return forces


def force_and_stress(constants, additions):
    # force of core in electric field
    force_ion = force_ion_from_external(
        constants.lattice_vectors,
        constants.number_ion_types,
        constants.ion_types,
        constants.number_ions,
        constants.ZVAL,
        constants.positions,
        constants.charge_density,
    )
    delta_force = force_ion 
    additions.forces += delta_force


def calc_dipole(chargedensity, Lz, nz, core_dipole, grid_position_frac =  grid_position_frac, grid_roll_frac = grid_roll_frac, dx_dipolcorr = 8):
    gridposition = int(grid_position_frac*nz)  
    grid_roll = int(grid_roll_frac*nz)  
    chargedensity_roll = chargedensity.copy()
    chargedensity_roll = np.roll(chargedensity_roll, grid_roll, axis = 2)
    dz = Lz/nz
    z = np.linspace(0, Lz, nz, endpoint = False)
    dipole = np.sum(np.mean(chargedensity_roll, (0,1))*z) * dz
    dipole -= core_dipole
    if(np.abs(dipole)<1e-5): return 0, chargedensity
    q = dipole / (dx_dipolcorr*dz)
    new_chargedensity = chargedensity.copy()
    new_chargedensity[:,:,gridposition] = q / dz
    new_chargedensity[:,:,gridposition+dx_dipolcorr] = -q / dz

    return dipole, new_chargedensity
    
def local_potential(constants, additions):
    rho = constants.charge_density
    nx, ny, nz = rho.shape
    box = constants.lattice_vectors
    Lx = box[0][0]
    Ly = box[1][1]
    Lz = box[2][2]
    dz = Lz/nz
    dV = Lx*Ly*Lx/nx/ny/nz
    Q_ext, V_ext = external_charge(nx, ny, nz, Q_list, rQ_list, box, mode='gaussianWall')
    # Q_ext has unit of el/angstrom^3
    # charge_density has unit of el (per grid)
    if not os.path.exists("Q_ext.npy"):
        np.save("Q_ext.npy", Q_ext)
    if not os.path.exists("V_ext.npy"):
        np.save("V_ext.npy", V_ext)
    total_charge = constants.charge_density - Q_ext*Lx*Ly*Lz
    total_nelect = np.sum(total_charge)/nx/ny/nz
    if check_normalization(nelect_neutral, total_nelect):
        # calculate dipole correction
        core_dipole = 0
        pos_cart = constants.positions @ constants.lattice_vectors
        for i in range(constants.number_ions):
            pos_z = (pos_cart[i, 2]+ int(grid_roll_frac*nz)*dz)%Lz
            core_dipole += constants.ZVAL[constants.ion_types[i]] * pos_z
        dipole, new_chargedensity = calc_dipole(total_charge/Lx/Ly/Lz, Lz, nz, core_dipole/Lx/Ly)
        total_pot = electrostatic_potential(new_chargedensity, Lx, Ly, Lz, epsilon_0=0.005526349358057108)
        delta_pot = total_pot - constants.hartree_potential 
    else:
        # first electronic step, NELECT = NELECT neutral
        delta_pot = V_ext
        dipole = 0

    np.save('delta_pot.npy', delta_pot)
    with open("dipole_corr.dat", "a") as file:  
        file.write(str(dipole*Lx*Ly)+"\n")
    additions.total_potential += delta_pot
 
    # correct the energy of core in external potential
    energy = 0
    for i in range(constants.number_ions):
        pos = constants.positions[i]
        pos_x = int(np.round(pos[0]*nx))%nx
        pos_y = int(np.round(pos[1]*ny))%ny
        pos_z = int(np.round(pos[2]*nz))%nz
        local_V = delta_pot[pos_x, pos_y, pos_z] 
        energy += constants.ZVAL[constants.ion_types[i]] * local_V
    additions.total_energy -= energy
    total_pot_z = np.mean(delta_pot + constants.hartree_potential + constants.ion_potential, axis = (0, 1))
       

def electrostatic_potential(rho, Lx, Ly, Lz, epsilon_0=0.005526349358057108):
    """
    Calculate the electrostatic potential from charge density in 3D with periodic boundary conditions.
    This version handles non-cubic (rectangular) grids.
    
    Parameters:
    - rho (3D numpy array): The charge density.
    - Lx, Ly, Lz (floats): The lengths of the periodic box in each direction.
    - epsilon_0 (float): The permittivity of free space (default is in SI units).
    
    Returns:
    - phi (3D numpy array): The electrostatic potential.
    """
    
    # Get the grid size in each dimension
    (Nx, Ny, Nz) = np.shape(rho)
    
    # Perform the Fourier transform of the charge density
    rho_k = np.fft.fftn(rho)
    
    # Create the wave vectors for the Fourier space (kx, ky, kz)
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
    kz = np.fft.fftfreq(Nz, d=Lz/Nz) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Compute k^2 = kx^2 + ky^2 + kz^2
    k_squared = kx**2 + ky**2 + kz**2
    
    # Solve for phi_k (handle the k = 0 mode separately)
    phi_k = np.zeros_like(rho_k)
    nonzero_mask = k_squared > 0  # Avoid division by zero for k = 0
    phi_k[nonzero_mask] = rho_k[nonzero_mask] / (epsilon_0 * k_squared[nonzero_mask])
    
    # Set the k = 0 mode to zero (this is the average charge in the system)
    phi_k[~nonzero_mask] = 0.0
    
    # Perform the inverse Fourier transform to get the potential in real space
    phi = np.fft.ifftn(phi_k).real
    
    return phi

        
    
        
    
    
    
