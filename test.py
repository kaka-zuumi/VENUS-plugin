import pluginplay as pp
from chemist import Atom, Molecule, ChemicalSystem, PointD, PointSetD
from simde import EnergyNuclearGradientStdVectorD
from friendzone.nwx2nwchem import NWChemEnergyAndGradientViaMolSSI, NWChemGradientViaMolSSI
#import nwchemex
from venus_nwchemex_plugin import BimolecularSamplingPT, UnimolecularSamplingPT, ImpactParameterSamplingPT, AngularMomentumThermalSamplingPT, VibrationalQuantaThermalSamplingPT, HessianAndFrequencyPT, OptimizerPT

import numpy as np
import random

################################################################################################################

# Some helper functions:

# Get the kinetic energy of the molecule
def getKineticEnergy(m,p):
    KE = 0.0
    for i in range(len(m)):
        KE += np.inner(p[i],p[i])/m[i]

    return 0.5e0 * KE

# Get the center-of-mass of the input molecule (not necessarily this whole molecule)
def getCenterOfMass(m,q):
    qCM = np.zeros((3))
    for i in range(len(m)):
        qCM += m[i]*q[i]

    return qCM / sum(m)

# Rotate a set of normal modes or velocities by a rotation matrix
def rotateNormalModes(nmodes,rotationMatrix):
    Nmodes = nmodes.shape[0]
    newmodes = np.copy(nmodes)
    for i in range(Nmodes):
        nmode = nmodes[i]
        for Natom in range(len(nmode)):
            newmodes[i][Natom] = np.dot(rotationMatrix.transpose(),nmode[Natom])

    return newmodes

# Calculate the moment of inertia tensor from the
# mass, m, and coordinates, q
def getMomentOfInertia(m,q):
    I = np.zeros((3,3))
    for i in range(len(m)):
        qi = q[i]
        I += m[i] * (np.diag(np.tile(np.inner(qi,qi),3)) - np.outer(qi,qi))

    return I

# Do an SVD to get the principal axes of a 3x3 matrix
def getPrincipalAxes(I):
    U, S, VH = np.linalg.svd(I)
    return S, U

# Choose a random rotation in 3D space
def chooseRandomSpatialRotation():

    # First, get a random unit vector for the x-axis
    while True:
        u = np.array([random.random(), random.random(), random.random()])
        u -= 0.5e0
        u2 = sum(u**2)
        if (u2 > 1.0e-4):
            u = u / np.sqrt(u2)
            break

    # Second, get a random unit vector for the y-axis
    # orthogonal to the already-chosen x-axis
    while True:
        v = np.array([random.random(), random.random(), random.random()])
        v -= 0.5e0
        v = np.cross(u,v)
        v2 = sum(v**2)
        if (v2 > 1.0e-4):
            v = v / np.sqrt(v2)
            break

    # Third, define the z-axis to be the unit vector
    # orthogonal to both the x- and y-axis, with the
    # sign chosen randomly
    w = np.cross(u,v)
    if (random.random() > 0.5e0):
        w = -w

    # Three orthogonal unit vectors are one way to
    # construct a random rotation matrix
    return np.array([u, v, w])

# Choose a random axis to spin the molecules, and also
# consider the principal axes if the molecule is linear
# Output the axis as a normed vector or "unit"
def chooseRandomUnitAngularVelocity(Nlinear,principalAxes):
    if (Nlinear):
        u = 2 * np.pi * random.random()
        a = principalAxes[0] * np.sin(u) + principalAxes[1] * np.cos(u)
    else:
        x = random.random()
        y = random.random()
        z = random.random()
        a = np.array([x,y,z])
    return a / np.norm(a)

# Calculate angular speed given the unit angular velocity,
# moment of inertia tensor, and the required amount of
# rotational energy
def getAngularSpeed(unitOmega,I,RE):
    REunit = np.inner(unitOmega,np.dot(I,unitOmega))
    return np.sqrt(2*RE/REunit)

# Calculate the angular momentum from the (relative)
# positions and momenta
def getAngularMomentum(q,p):
    L = np.zeros((3))
    for i in range(len(q)):
        L += np.cross(q[i],p[i])

    return L

# Assuming the system is rigid, return the linear momenta
# which correspond to the given input angular velocity
def getMomentaFromAngularVelocity(m,q,omega):
    p = np.zeros(np.shape(q))
    for i in range(len(m)):
        p[i] = m[i]*np.cross(omega,q[i])

    return p

# Calculate the rotational energy and angular velocity, 
# omega, of the system
def getErotAndOmegaFromL(m,q,L):
    I = getMomentOfInertia(m,q)

    # The singular value decomposition will be able to
    # ignore essentially-zero components of the moment
    # of inertia tensor (e.g., for linear molecules) and
    # allow the inverse to be taken easily
    U, S, VH = np.linalg.svd(I)
    Sinv = np.diag(S)
    for i in range(3):
        if (Sinv[i,i] > 1.0e-12):
            Sinv[i,i] = 1.0e0 / Sinv[i,i]
        else:
            Sinv[i,i] = 0.0e0

    omega = np.dot(VH.transpose() @ Sinv @ U.transpose(), L)
    Erot = 0.5e0*np.dot(omega,L)

    return Erot, omega

# Calculate the rotational energy and angular velocity, 
# omega, of the system
def getErotAndOmegaFromP(m,q,p):
    L = getAngularMomentum(q,p)
    Erot, omega = getErotAndOmegaFromL(m,q,L)

    return Erot, omega



# Separate the two molecules (dynamicChemicalSystems)
def separateMolecules(molecule1, molecule2):
    dSEPARATED = 400.0   # Distance to separate molecules by

    masses1 = molecule1.masses
    masses2 = molecule2.masses
    q1 = molecules1.positions
    q2 = molecules2.positions
    r1 = getCenterOfMass(masses1,q1)
    r2 = getCenterOfMass(masses2,q2)

    for i in range(molecule1.Natoms):
        q1[i] -= r1
        q1[i][0] += dSEPARATED
    molecule1.set_positions(q1)

    for i in range(molecule2.Natoms):
        q2[i] -= r2
        q2[i][0] -= dSEPARATED
    molecule2.set_positions(q2)



# Calculate the vibrational energy contribution of each
# normal mode, and then compute the amplitude of that
# mode's harmonic-oscillator-like spatial displacement
def getVibrationalEnergiesAndAmplitudes(freqs,vibNums):

    Nmodes = len(freqs)
    Evibs = []
    amplitudes = []
    for i in range(Nmodes):
        Evib = (0.5+vibNums[i])*freqs[i]
        Evibs.append(Evib)
        amplitudes.append(np.sqrt(2*Evib)/freqs[i])

    return np.array(Evibs), np.array(amplitudes)


# Set the positions of atoms in this molecule
def chemical_system_set_positions(chem_sys,q):

    Zs = []
    atomnames = []
    masses = []
    Natoms = 0
    for atom in chem_sys.molecule:
        atomnames.append(atom.name)
        Zs.append(atom.Z)
        masses.append(atom.mass)
        Natoms += 1

    assert(len(q) == Natoms)

    chemist_mol = Molecule()
    for i in range(Natoms):
        chemist_mol.push_back(Atom(
               atomnames[i],Zs[i],masses[i],*q[i]))
    chemist_mol.set_charge(chem_sys.molecule.charge())
    chemist_mol.set_multiplicity(chem_sys.molecule.multiplicity())

    return ChemicalSystem(chemist_mol)


################################################################################################################

# Some helper functions to aid with the semiclassical sampling

# Calculate Gauss-Legendre x-values and weights (x,w(x))
def gaussLegendre(xLEFT,xRIGHT,n,xx=None):

    # First, get the roots of the Gauss-Legendre
    # polynomial, if they were not given
    m = int((n+1)/2)
    if (xx == None):
        xx = np.zeros((m))
        df = np.zeros((m))
        for i in range(m):
            j = i + 1
            xx[i] = np.cos(np.pi*(j - 0.25e0)/(n+0.50e0))

            # Find the root with the
            # Newton-Raphson formula
            f = 1.0e0
            while (abs(f) > 1.0e-13):
                f, df[i] = legendrePolynomial(xx[i],n)
                xx[i] = xx[i] - f / df[i]

    # Finally, get the points (x,w(x)) at
    # which the polynomial is evaluated

    x = np.zeros((n))
    w = np.zeros((n))

    xm = 0.5e0 * (xRIGHT + xLEFT)
    xl = 0.5e0 * (xRIGHT - xLEFT)
    for i in range(m):
        x[i]   = xm - xl * xx[i]
        x[(n-1)-i] = xm + xl * xx[i]
        w[i]   = (xl / ((1.0e0 - xx[i]*xx[i]) * df[i] * df[i]) ) / 0.5e0
        w[(n-1)-i] = w[i]

    return x, w

# Calculate the legendre polynomial of
# order n at x, and its derivative
def legendrePolynomial(x,n):
    p2 = x
    pl = 1.5e0 * x * x - 0.5e0
    for k in range(2,n):
        p1 = p2
        p2 = pl
        pl = float((2*k+1) * p2 * x - k * p1) / (k+1)

    dpl = n * (x * pl - p2) / (x * x - 1.0e0)

    return pl, dpl



################################################################################################################

# A new class to stick a ChemicalSystem and momenta (+ helper functions) together

class dynamicChemicalSystem:
    def __init__(self,chemical_system,momenta=None,eANDgMOD=None):
        self.chemical_system = chemical_system
        self.Zs = []
        self.atomnames = []
        self.masses = []
        self.positions = []
        self.Natoms = 0
        for atom in self.chemical_system.molecule:
            self.atomnames.append(atom.name)
            self.Zs.append(atom.Z)
            self.masses.append(atom.mass)
            self.positions.append([atom.x, atom.y, atom.z])
            self.Natoms += 1

        self.charge = self.chemical_system.molecule.charge()
        self.multiplicity = self.chemical_system.molecule.multiplicity()
        self.masses = np.array(self.masses)
        self.positions = np.array(self.positions)

        # Set the momenta (and velocities)
        if (momenta is None):
            momenta = np.zeros((self.Natoms,3))
        self.velocities = np.zeros((self.Natoms,3))
        self.set_momenta(momenta)

        # Set up the energy + gradient calculator
#       if (eANDgMOD):
#           self.eANDgMOD = eANDgMOD
        self.eANDgMOD = eANDgMOD

    # Set the positions of atoms in this molecule
    def set_positions(self,q):
        assert(len(q) == self.Natoms)

        self.positions = q

        chemist_mol = Molecule()
        for i in range(self.Natoms):
            chemist_mol.push_back(Atom(
                   self.atomnames[i],self.Zs[i],self.masses[i],
                   *self.positions[i]))
        chemist_mol.set_charge(self.charge)
        chemist_mol.set_multiplicity(self.multiplicity)

        self.chemical_system = ChemicalSystem(chemist_mol)

    # Set the momenta of atoms in this molecule
    def set_momenta(self,p):
        assert(len(p) == self.Natoms)

        self.momenta = p
        for i in range(self.Natoms):
            self.velocities[i] = self.momenta[i] / self.masses[i]

    # Set the velocities of atoms in this molecule
    def set_velocities(self,v):
        assert(len(v) == self.Natoms)

        self.velocities = v
        for i in range(self.Natoms):
            self.momenta[i] = self.velocities[i] * self.masses[i]

    # Place the center-of-mass of the molecule at the origin
    def centerMolecule(self):
        q = np.copy(self.positions)
        r = getCenterOfMass(self.masses,q)
        for i in range(self.Natoms):
            q[i] -= r

        self.set_positions(q)

    # Rotate the molecule's coordinates by a rotation matrix
    def rotateMolecule(self,rotationMatrix):
        q = np.copy(self.positions)
        p = np.copy(self.momenta)
        for i in range(self.Natoms):
            q[i] = np.dot(rotationMatrix.transpose(),q[i])
            p[i] = np.dot(rotationMatrix.transpose(),p[i])

        self.set_positions(q)
        self.set_momenta(p)

    # After choosing an angular momentum, the frequencies,
    # normal modes, and vibrational quanta, choose a set
    # of initial coordinates
    def chooseQPgivenNandLandNormalModes(self,vibNums,L0,freqs,nmodes):

        debug = True

        # Get the energy at this optimized structure;
        # this MUST be a energy minimum for this program
        # to work smoothly
#       self.separateMolecules()
        self.centerMolecule()
        eANDg = self.eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),self.chemical_system,PointSetD())
        q0 = self.positions; E0 = eANDg.pop(-1); g0 = np.array(eANDg).reshape(-1,3)
        if (debug): print("E0:",E0, "q0:", q0)

        # Now, get the corresponding vibrational and
        # rotational energies
        Evibs, amplitudes = getVibrationalEnergiesAndAmplitudes(freqs,vibNums)
        Evib0 = sum(Evibs)
        Erot0, omega0 = getErotAndOmegaFromL(self.masses,self.positions,L0)

        Eint0 = Evib0 + Erot0
        if (debug): print("Evib0,Erot0,Eint0:",Evib0,Erot0,Eint0)

        Nmodes = len(freqs)
        NpureKinetic=0

        dq = np.zeros((Nmodes))
        dp = np.zeros((Nmodes))
        while True:

            # Fill up some normal modes with purely
            # kinetic energy
            for i in range(NpureKinetic):
                dq[i] = 0.0e0
                dp[i] = -freqs[i]*amplitudes[i]

            # Fill up the other normal modes with a mix
            # of kinetic and potential energy
            for i in range(NpureKinetic,Nmodes):
                u = 2 * np.pi * random.random()
                dq[i] = amplitudes[i]*np.cos(u) / (8065.5401*1.8836518e-3)
                dp[i] = -freqs[i]*amplitudes[i]*np.sin(u)

            # Modify the original optimized structure
            # with the perturbations
            q = np.copy(q0)
            p = np.zeros(np.shape(q))
            for i in range(Nmodes):
                q += nmodes[i] * dq[i]
                p += nmodes[i] * dp[i]

            for i in range(self.Natoms):
                p[i] = p[i] * self.masses[i]

            self.set_positions(q)
            self.set_momenta(p)
            qCM = getCenterOfMass(self.masses,q)

            # Calculate the new (probably different)
            # angular momentum
            L = getAngularMomentum(q-qCM,p)

            # Comput the "difference" in angular
            # momentum and velocity
            Ldiff = L - L0
            Erot, omega = getErotAndOmegaFromL(self.masses,q-qCM,Ldiff)

            # Use this angular momentum difference to
            # adjust linear momenta for an accurate 
            # angular momentum and rotational energy
            pdiff = getMomentaFromAngularVelocity(self.masses,q-qCM,omega)
            p -= pdiff

            # Compute the new total potential and kinetic
            # energies of the system

            qBOTH = np.copy(q)
            self.set_positions(qBOTH)
            KE = getKineticEnergy(self.masses,p)

#           self.separateMolecules()
            eANDg = self.eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),self.chemical_system,PointSetD())
            E = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)
            if (debug): print("E:",E, "q:",self.positions)
            E = E + KE
            self.centerMolecule()

            # The internal energy of the molecule of interest
            # will be the difference between the total energy
            # and the reference energy from earlier (when it
            # is at the local minimum)
            Eint = E - E0

            # See how close this internal energy is to the
            # required amount
            fitnessOfEint = abs(Eint0-Eint)/Eint0

            qCM = getCenterOfMass(self.masses,q)
            L = getAngularMomentum(q-qCM,p)
            Erot, omega = getErotAndOmegaFromL(self.masses,q-qCM,L)

            if (debug):
                print("KErot: ", Erot, " KEvib: ", KE-Erot)
                print(NpureKinetic,fitnessOfEint,Eint0,Eint)

            # First, if all of the internal energy is rotational,
            # then do not attempt any scaling and just exit
            if (Evib0 < 1.0e-4*Eint0): break

            # If the internal energy is not that close, then just
            # convert one of the normal modes to being purely
            # kinetic
            if (fitnessOfEint >= 0.1e0 and NpureKinetic < Nmodes):
                NpureKinetic += 1
                continue

            # If the internal energy is close to that required,
            # try scaling it
            Nscale = 0
            #while (Nscale < 1000 and fitnessOfEint >= 0.001e0):
            while (Nscale < 50 and fitnessOfEint >= 0.001e0):
                scalingFactor = np.sqrt(Eint0/Eint)
                p = p * scalingFactor
                q = q0 + (q - q0) * scalingFactor

                qCM = getCenterOfMass(self.masses,q)

                L = getAngularMomentum(q-qCM,p)

                Ldiff = L - L0
                Erot, omega = getErotAndOmegaFromL(self.masses,q-qCM,Ldiff)

                pdiff = getMomentaFromAngularVelocity(self.masses,q-qCM,omega)
                p -= pdiff

                qBOTH = np.copy(q)
                self.set_positions(qBOTH)
                KE = getKineticEnergy(self.masses,p)

#               self.separateMolecules()
                eANDg = self.eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),self.chemical_system,PointSetD())
                E = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)
                if (debug): print("E:",E, "q:",self.positions)
                E = E + KE
                self.centerMolecule()

                Eint = E - E0

                fitnessOfEint = abs(Eint0-Eint)/Eint0

                qCM = getCenterOfMass(self.masses,q)
                L = getAngularMomentum(q-qCM,p)
                Erot, omega = getErotAndOmegaFromL(self.masses,q-qCM,L)

                if (debug):
                    print("Erot:",Erot)
                    print(fitnessOfEint,Eint0,Eint)
                    print(q)

                if (fitnessOfEint >= 0.1e0 and NpureKinetic < Nmodes):
                    NpureKinetic += 1
                    break

                Nscale += 1

            if (fitnessOfEint < 0.001e0): break

        pBOTH = np.copy(p)
        self.set_momenta(pBOTH)


    # Sample a diatomic molecule's acceptable range
    # of bond lengths for the specified quantum numbers
    def getDiatomBondLengthRangeWithEBK(self,Nrot0,Nvib0,Evib0,V0):

        debug = True

        # Some conversion constant
        rotConstant2energy = 0.00209008   # eV

        # Threshold for error in rovibrational energy
        # (in quanta) for the semiclassical sampling
        Nvib_error_threshold = 1.0e-1 # 1.0e-5

        # Initialize some information about the diatom
        m = self.masses
        mu  = m[0] * m[1] / (m[0] + m[1])
        q = self.positions
        bondLength = np.sqrt(sum((q[1]-q[0])**2))

        hnu = Evib0 / (Nvib0 + 0.5e0)
        AM0  = (np.sqrt(2 * rotConstant2energy)) * np.sqrt(float(Nrot0*(Nrot0+1)))

        if (debug):
            print("AM from EBK:", AM0)
            print("R0: ", bondLength, "I=mu*R0^2: ", mu*(bondLength**2), "Erot: ", (AM0**2)/(2*mu*(bondLength**2)))

        # Only accept the range of bond lengths
        # found by the algorithm "FINLNJ" if its
        # expected vibrational quantum number
        # agrees with the requested one
        Ntries = 0
        Erovib = Evib0
        Nvib_error = 1.0e0
        while (abs(Nvib_error) > Nvib_error_threshold):
            Nvib, rMIN, rMAX = self.getRovibrationalRange(Erovib,AM0,V0)
            Nvib_error = Nvib0 - Nvib
            Erovib = Erovib + Nvib_error * hnu

            if (debug): print("Nvib0:",Nvib0, "Nvib:",Nvib, "Nvib_error:",Nvib_error)

            Ntries += 1
            if (Ntries > 200):
                raise ValueError("In getDiatomBondLengthRangeWithEBK: Ntries for diatom sampling above 200")

        # Make sure not to include the turning points themselves
        # in the subsequent distance scans
        rMIN += 0.001e0
        rMAX -= 0.001e0

        pTEST = np.sqrt(2 * mu * Erovib * 0.0001e0)

        return rMIN, rMAX, AM0, Erovib, pTEST

    # Find out the "turning points" or the bond
    # lengths which are accessible with this
    # vibrational and rotational energy
    def getRovibrationalRange(self,E,angularMomentum,V0):

        debug = True

        # A larger number of points will make the
        # semiclassical sampling converge smoother
        NorderPolynomial = 50    # 100

        # Some conversion constant
        rotConstant2energy = 0.00209008   # eV

        # Initialize some information about the diatom
        L2 = angularMomentum**2
        m = self.masses
        mu  = m[0] * m[1] / (m[0] + m[1])
        qBOTH = self.positions
        q = np.copy(qBOTH)
        bondLength = np.sqrt(sum((q[1]-q[0])**2))

        # If the bond length at the energy minimum is too
        # large, print this warning
        if (bondLength > 6):
            raise ValueError("In FINLNJ: bondLength > 6 for diatom sampling")

        # Position the diatom so that the bond is along
        # the z-axis; place the atoms so that (assuming
        # the two reactants started separate) the two
        # reactants stay separate
        qCM = getCenterOfMass(m,q)
        q[0] = qCM
        q[0][2] -= 0.5e0 * bondLength
        q[1] = qCM
        q[1][2] += 0.5e0 * bondLength

        # Calculate the energy; only do this when the two
        # reactants are separate
        qBOTH = np.copy(q)
        self.set_positions(qBOTH)
        eANDg = self.eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),self.chemical_system,PointSetD())
        V = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)
        if (debug): print("V:",V)

        # Calculate Veff, the rotational + vibrational
        # energy
        Erot0 = L2 / (2 * mu * (bondLength**2))
        Veff = (V - V0) + Erot0

        if (debug):
            print("In FINLNJ: V-V0 = ", (V-V0), "E(not V-V0) = ", Erot0)
            print("In FINLNJ: Veff = ", Veff, "   E = ", E)

        # If the energy is ALREADY above that required
        # for the system, then print this warning
        if (Veff > E):
            raise ValueError("In FINLNJ: Veff > E for diatom sampling")

        # If there are no problems, go on ahead with
        # integrating the energy

        # First, define the boundaries of the integral
        # as the bond lengths which just are just 
        # barely going over the required energy E

        # Find the upper bound
        q[1][2] = q[0][2] + bondLength
        while True:

            # Increment the bond by values of 0.001 A
            q[1][2] += 0.001e0

            # Evaluate the new energy and Veff
            qBOTH = np.copy(q)
            self.set_positions(qBOTH)
            eANDg = self.eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),self.chemical_system,PointSetD())
            V = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)

            newBondLength = q[1][2] - q[0][2]
            Veff = (V - V0) + L2 / (2 * mu * (newBondLength**2))

            # Eventually, the potential energy should
            # rise enough to get above the threshold
            if (Veff >= E or newBondLength > 50):
                break

        rMAX = newBondLength
        if (debug): print("In FINLNJ: rMAX(A) = ", rMAX)

        # Find the lower bound
        q[1][2] = q[0][2] + bondLength
        while True:

            # Decrement the bond by values of 0.001 A
            q[1][2] -= 0.001e0

            # Evaluate the new energy and Veff
            qBOTH = np.copy(q)
            self.set_positions(qBOTH)
            eANDg = self.eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),self.chemical_system,PointSetD())
            V = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)

            # Eventually, the potential energy should
            # rise enough to get above the threshold
            newBondLength = q[1][2] - q[0][2]
            Veff = (V - V0) + L2 / (2 * mu * (newBondLength**2))

            if (Veff >= E):
                break

        rMIN = newBondLength
        if (debug): print("In FINLNJ: rMIN(A) = ", rMIN)

        # Prepare the bond lengths for which to
        # integrate the energy over

#       r, w = GLPAR(rMIN,rMAX,NorderPolynomial)
        r, w = self.gaussLegendre(rMIN,rMAX,NorderPolynomial)

        # Next, take the integral by evaluating the
        # energy on these points
        Asum = 0.0e0
        for j in range(NorderPolynomial):
            newBondLength = r[j]

            q[1][2] = q[0][2] + newBondLength
            qBOTH = np.copy(q)
            self.set_positions(qBOTH)
            eANDg = self.eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),self.chemical_system,PointSetD())
            V = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)

            Veff = (V - V0) + L2 / (2 * mu * (newBondLength**2))

            # If some of the points are above the
            # integral (should only be the endpoints)
            # don't add them into the integral
            if (E > Veff):
                Asum += w[j] * np.sqrt(E-Veff)
            else:
                if (debug): print("In FINLNJ: see point with E<Veff (r(A),E(kcal/mol),Veff(kcal/mol)) = ", (newBondLength, E, Veff))

        if (debug): print("In FINLNJ: Integral = ", Asum)

        # Return the molecule to its minimum
        # energy configuration for now
        q[1][2] = q[0][2] + bondLength
        qBOTH = np.copy(q)
        self.set_positions(qBOTH)

        # The "expected" vibrational quantum number
        # corresponding to this energy can then be
        # computed to see if it agrees with the
        # value requested
        Nvib = np.sqrt(8.0e0 * mu) * Asum / (2*np.pi* np.sqrt(2*rotConstant2energy))
        Nvib = Nvib - 0.5e0

        return Nvib, rMIN, rMAX




################################################################################################################

# Modules to help with the unimolecular sampling

class numericalHessianAndFrequencyAnalysis(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(HessianAndFrequencyPT())
        self.description("Analyze frequencies and normal modes of a chemical system by computing the hessian")
        self.add_input("numerical derivative geometry step size")
        self.add_input("keywords").set_default({})
        self.add_submodule(EnergyNuclearGradientStdVectorD(),'gradient submodule')

    def run_(self, inputs, submods):
        pt = HessianAndFrequencyPT()
        mol0, = pt.unwrap_inputs(inputs)
        stepSize = inputs['numerical derivative geometry step size'].value()
        gMOD = submods["gradient submodule"]

        q = []; masses = []; Natoms = 0
        for atom in mol0.molecule:
            q.append([atom.x, atom.y, atom.z])
            masses.append(atom.mass)
            Natoms += 1
        q = np.array(q); masses = np.array(masses)
        print("GOT HERE")

        Nsteps = 0
        mol = chemical_system_set_positions(mol0,q)
        g = gMOD.run_as(EnergyNuclearGradientStdVectorD(),mol,PointSetD())
        g = np.array(g)
        q0 = np.copy(q); g0 = np.copy(g)

        hessian = []; Nmodes = 0
#       for j in range(3):
#           for i in range(Natoms):
        for i in range(Natoms):
            for j in range(3):
                print("why", i, j, q)
                q = np.copy(q0); q[i][j] = q0[i][j] + stepSize
                mol = chemical_system_set_positions(mol0,q)
                g = gMOD.run_as(EnergyNuclearGradientStdVectorD(),mol,PointSetD())
                print("boo", i, j)
                gPLUS = np.array(g)

                print("bug", i, j)
                q = np.copy(q0); q[i][j] = q0[i][j] - stepSize
                del mol, g
                mol = chemical_system_set_positions(mol0,q)
                print("kkk", i, j)
                g = gMOD.run_as(EnergyNuclearGradientStdVectorD(),mol,PointSetD())
                print("man", i, j)
                gMINUS = np.array(g)
                print("huh", i, j)

                hessian.append((gPLUS-gMINUS)/(2*stepSize))
#               hessian.append((gPLUS-g0)/(stepSize))
                Nmodes += 1
        hessian = np.array(hessian)
        print("whatttt")

        massSQRTinv = np.repeat(masses**-0.5, 3)
        hbar_SI = 1.0545718001391127e-34
        meter_in_A = 1.0e10
        e_SI = 1.6021766208e-19
        dalton_SI = 1.66053904e-27
#       unit_conversion = hbar_SI * meter_in_A / np.sqrt(e_SI * dalton_SI)
        unit_conversion = hbar_SI * meter_in_A / (3 * np.sqrt(e_SI * dalton_SI))   # Very odd that we must divide by 3...

        Ha_to_invcm = 219474.63

        # Diagonalize!
        eigenvalues, eigenvectors = np.linalg.eigh(massSQRTinv*hessian*massSQRTinv[:, np.newaxis])
#       eigenvalues, eigenvectors = np.linalg.eigh(massSQRTinv*hessian.T*massSQRTinv[:, np.newaxis])

        for i, eigenvalue in enumerate(eigenvalues):
            print("eigenvalue:", i, eigenvalues[i])
            if np.iscomplex(eigenvalue):
                eigenvalues[i] = -np.sqrt(np.linalg.norm(eigenvalue))
            elif (eigenvalue <= 0):
                eigenvalues[i] = -np.sqrt(-eigenvalue)
            else:
                eigenvalues[i] = np.sqrt(eigenvalue)
        index_order = eigenvalues.argsort()

        frequencies = []; normal_modes = []
        for i in index_order:
            frequencies.append(eigenvalues[i] * unit_conversion * Ha_to_invcm)
            normal_modes.append(eigenvectors[i])
        normal_modes = np.array(normal_modes)
        normal_modes = normal_modes.reshape(3*Natoms,Natoms,3) * masses[np.newaxis, :, np.newaxis]**-0.5

        ######################################################################################
        # Print out the frequencies
    
        if (True):
            print("\nNWChemEx molecule:")
            print(mol0.molecule)
            print("NWChemEx frequencies:")
            Nmode_stride = 6
            Nstrides = int(1 + np.floor((Nmodes-1) * 1.0 / Nmode_stride))
        
            atomnames = []; Natoms = 0
            for atom in mol0.molecule:
                atomnames.append(atom.name)
                Natoms += 1
        
            for Nstride in range(Nstrides):
                indexes = range(Nstride*Nmode_stride, min((Nstride+1)*Nmode_stride,Nmodes))
                print("")
                print("{0:18s}   ".format("Frequencies:") + " ".join(["{0:12.2f}".format(frequencies[index]) for index in indexes]))
                print("")
                for Natom in range(Natoms):
                    for i, xyzID in enumerate(["x","y","z"]):
                        print("{0:9s} {1:3d} {2:2s} {3:1s}   ".format(" . ",Natom,atomnames[Natom],xyzID) + " ".join(["{0:12.6f}".format(normal_modes[index][Natom][i]) for index in indexes]))
    
        ######################################################################################

        normal_modes = list(normal_modes.flatten())

        rv = self.results()
        return pt.wrap_results(rv, frequencies, normal_modes)


class simpleGradientDescentLineSearch(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(OptimizerPT())
        self.description("Optimize a chemical system")
        self.add_input("default step size")
        self.add_input("gradient convergence threshold")
        self.add_input("keywords").set_default({})
        self.add_submodule(EnergyNuclearGradientStdVectorD(),'energy and gradient submodule')

    def run_(self, inputs, submods):
        pt = OptimizerPT()
        mol0, = pt.unwrap_inputs(inputs)
        stepSize0 = inputs['default step size'].value()
        grad_threshold = inputs['gradient convergence threshold'].value()
        eANDgMOD = submods["energy and gradient submodule"]

        q = []
        for atom in mol0.molecule:
            q.append([atom.x, atom.y, atom.z])
        q = np.array(q)

        Nsteps = 0
        mol = chemical_system_set_positions(mol0,q)
        eANDg = eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),mol,PointSetD())
        e = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)
        q0 = q; e0 = e; stepSize = stepSize0
        g_norm = np.linalg.norm(g)
        print("OPT |dEdq| threshold: {0:12.4f}".format(grad_threshold))
        print("OPT STEP: {0:4d}   E: {1:18.10f}   |dEdq|: {2:12.4f}".format(Nsteps,e,g_norm))
        while (g_norm > grad_threshold):
            dq = stepSize*g
            dqMAX = np.max(np.abs(dq))
            if (dqMAX > 0.5): dq = dq * 0.5 / dqMAX
            q -= dq
            mol = chemical_system_set_positions(mol,q)
            eANDg = eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),mol,PointSetD())
            e = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)
            g_norm = np.linalg.norm(g)
            Nsteps += 1
            print("OPT STEP: {0:4d}   E: {1:18.10f}   |dEdq|: {2:12.4f}".format(Nsteps,e,g_norm))

            if (e0 < e):
                stepSize = stepSize * 0.5
                q = q0; e = e0
            else:
                setSize = stepSize0
            q0 = q; e0 = e

        print("OPT CONVERGED!")
        rv = self.results()
        return pt.wrap_results(rv, mol)

################################################################################################################

# Unimolecular sampling modules

class unimolecularThermalSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(UnimolecularSamplingPT())
        self.description("Do a unimolecular thermal sampling")
        self.add_input("rotational sample")
        self.add_input("vibrational sample")
        self.add_input("keywords").set_default({})
        self.add_submodule(EnergyNuclearGradientStdVectorD(),'energy and gradient submodule')
        self.add_submodule(HessianAndFrequencyPT(),'frequency submodule')
        self.add_submodule(OptimizerPT(),'optimization submodule')
        self.add_submodule(VibrationalQuantaThermalSamplingPT(),'vibration sampling submodule')
        self.add_submodule(AngularMomentumThermalSamplingPT(),'linear rotation sampling submodule')
        self.add_submodule(AngularMomentumThermalSamplingPT(),'nonlinear rotation sampling submodule')

    def run_(self, inputs, submods):
        pt = UnimolecularSamplingPT()
        mol0, = pt.unwrap_inputs(inputs)
        Trot = inputs['rotational sample'].value()
        Tvib = inputs['vibrational sample'].value()
        keywords = inputs['keywords'].value()
        eANDgMOD = submods["energy and gradient submodule"]
        optMOD = submods["optimization submodule"]
        freqMOD = submods["frequency submodule"]
        vibMOD = submods["vibration sampling submodule"]
        linrotMOD = submods["linear rotation sampling submodule"]
        nonlinrotMOD = submods["nonlinear rotation sampling submodule"]

        debug = True

        # Some conversion constants
        kB = 8.617281e-5                  # eV
        freq2energy = 1.000               # cm^-1

        # Check the geometry is a stationary point and
        # get its frequencies
        optimized_chemist_sys = optMOD.run_as(OptimizerPT(), mol0)
        frequencies, normal_modes = freqMOD.run_as(HessianAndFrequencyPT(), optimized_chemist_sys)
    
        Nmodes = len(frequencies)
        frequencies = np.array(frequencies)
        normal_modes = np.array(normal_modes).reshape(Nmodes,-1,3)

        # Start getting the momenta and other dynamic
        # properties of the system ready
        chemist_sys = dynamicChemicalSystem(optimized_chemist_sys,eANDgMOD=eANDgMOD)

        I = getMomentOfInertia(chemist_sys.masses,chemist_sys.positions)
        axesMasses, axesVectors = getPrincipalAxes(I)
        print("I:", I, "\nprincipal moments of inertia:", axesMasses)

        # Determine whether the molecules is linear by looking at the smallest principal axis
        # Then use this sample to sample an angular momentum vecotr
        if (axesMasses[2] < axesMasses[1]*1.0e-12):
            linear = True
            nonzero_frequencies = np.real(frequencies[5:]) * freq2energy
            nonzero_normal_modes = normal_modes[5:]
#           Erot, L = linrotMOD.run_as(AngularMomentumThermalSamplingPT(), list(axesMasses))
            Erot, L = linrotMOD.run_as(AngularMomentumThermalSamplingPT(), PointD(*axesMasses))
            L = [L.x, L.y, L.z]
        else:
            linear = False
            nonzero_frequencies = np.real(frequencies[6:]) * freq2energy
            nonzero_normal_modes = normal_modes[6:]
#           Erot, L = nonlinrotMOD.run_as(AngularMomentumThermalSamplingPT(), list(axesMasses))
            Erot, L = nonlinrotMOD.run_as(AngularMomentumThermalSamplingPT(), PointD(*axesMasses))
            L = [L.x, L.y, L.z]

        if (debug): print("Reactant is linear?", linear)

        # Orient the molecule and its normal modes in the principal axes coordinates
        chemist_sys.rotateMolecule(axesVectors)
        normal_modes = rotateNormalModes(normal_modes,axesVectors)
        nonzero_normal_modes = rotateNormalModes(nonzero_normal_modes,axesVectors)

        # Sample the vibrational quanta assuming uncoupled oscillators
        vibNums = []
        for nonzero_freq in nonzero_frequencies:
            vibNums.append(vibMOD.run_as(VibrationalQuantaThermalSamplingPT(), nonzero_freq))

        if (debug):
            print("kBTrot:",kB*Trot)
            print("kBTvib:",kB*Tvib)
            print("nonzeroFreqs:",nonzero_frequencies)
            print("nonzeroModes:",nonzero_normal_modes)
            print("Erot:",Erot)
            print("L:", L)
            print("Nvib:",vibNums)

        # Use these variables to pick initial coordinates Q
        # and momenta P for the single molecule
        chemist_sys.chooseQPgivenNandLandNormalModes(vibNums,L,nonzero_frequencies,nonzero_normal_modes)

        if (debug):
            ErotTEST, omegaTEST = getErotAndOmegaFromP(chemist_sys.masses,
                                         chemist_sys.positions,chemist_sys.momenta)
            print("Erot:",ErotTEST)
            print("omega:",omegaTEST)

        # After choosing relative positions, rotate the
        # molecule randomly in 3D space
        axesVectors = chooseRandomSpatialRotation()
        chemist_sys.centerMolecule()
        chemist_sys.rotateMolecule(axesVectors)

        sampled_momenta = PointSetD()
        for momentum in chemist_sys.momenta:
            sampled_momenta.push_back(PointD(*momentum))

        rv = self.results()
        return pt.wrap_results(rv, chemist_sys.chemical_system, sampled_momenta)


class unimolecularSemiclassicalSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(UnimolecularSamplingPT())
        self.description("Do a unimolecular semiclassical sampling (for diatomics)")
        self.add_input("rotational sample")
        self.add_input("vibrational sample")
        self.add_input("keywords").set_default({})
        self.add_submodule(EnergyNuclearGradientStdVectorD(),'energy and gradient submodule')
        self.add_submodule(HessianAndFrequencyPT(),'frequency submodule')
        self.add_submodule(OptimizerPT(),'optimization submodule')

    def run_(self, inputs, submods):
        pt = UnimolecularSamplingPT()
        mol0, = pt.unwrap_inputs(inputs)
        Nrot = inputs['rotational sample'].value()
        Nvib = inputs['vibrational sample'].value()
        keywords = inputs['keywords'].value()
        eANDgMOD = submods["energy and gradient submodule"]
        optMOD = submods["optimization submodule"]
        freqMOD = submods["frequency submodule"]

        # Make sure the inputs (although floats) are close enough to integers
        assert(Nrot == int(Nrot)); assert(Nvib == int(Nvib))
        Nrot = int(Nrot); Nvib = int(Nvib)

        debug = True

        # Some conversion constants
        kB = 8.617281e-5                  # eV
        freq2energy = 1.000               # cm^-1

        # Check the geometry is a stationary point and
        # get its frequencies
        optimized_chemist_sys = optMOD.run_as(OptimizerPT(), mol0)
        frequencies, normal_modes = freqMOD.run_as(HessianAndFrequencyPT(), optimized_chemist_sys)
    
        Nmodes = len(frequencies)
        frequencies = np.array(frequencies)
        normal_modes = np.array(normal_modes).reshape(Nmodes,Nmodes)

        # There's only one frequency
        freq = frequencies[-1]

        # Start getting the momenta and other dynamic
        # properties of the system ready
        chemist_sys = dynamicChemicalSystem(optimized_chemist_sys,eANDgMOD=eANDgMOD)
        p = chemist_sys.momenta


        m = chemist_sys.masses
        mu  = m[0] * m[1] / (m[0] + m[1])
        q = np.copy(chemist_sys.positions)
        qCM = getCenterOfMass(m,q)

        if (debug): print("Nvib:", [Nvib])

        # Get the energy at this optimized structure;
        # this MUST be a energy minimum for this program
        # to work smoothly
#       self.separateMolecules()
        eANDg = eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),chemist_sys.chemical_system,PointSetD())
        V0 = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)
        if (debug): print("V0:",V0)

        # Get the "turning points" or the bounds for
        # the bond length
        Evib = freq * (Nvib + 0.5e0)
        rMIN, rMAX, AM, Erovib, pTEST = chemist_sys.getDiatomBondLengthRangeWithEBK(Nrot,Nvib,Evib,V0)

        if (debug):
            print("Evib: ", Evib, "Erovib: ", Erovib)
            print("Rmin: ", rMIN, "Rmax: ", rMAX)

        ErotR2 = (AM**2) / (2*mu)

        # Try out a lot of different positions for the
        # atoms in space; stop when the energy is
        # nearly correct
        while True:
            u = random.random()
            r = rMIN + (rMAX - rMIN) * u

            q[0] = qCM
            q[0][2] -= 0.5e0 * r
            q[1] = qCM
            q[1][2] += 0.5e0 * r

            qBOTH = np.copy(q)
            chemist_sys.set_positions(qBOTH)
            eANDg = eANDgMOD.run_as(EnergyNuclearGradientStdVectorD(),chemist_sys.chemical_system,PointSetD())
            V = eANDg.pop(-1); g = np.array(eANDg).reshape(-1,3)

            Vdiff = (V - V0)
            Ediff = Erovib - ((ErotR2 / (r**2)) + Vdiff)

            # This case should occur very rarely if at all...
            # make sure to set Rmin and Rmax correctly to avoid this
            # (so that the MC sampling is correct)
            if (Ediff <= 0.0e0):
                if (self.debug): print("initQP diatom iteration.... ACCEPTED for (r,Ediff) = ", (r,Ediff))
                Ediff = 0.0e0
                PR = 0.0e0
                break

            else:
                PR = np.sqrt(2.0e0*mu*Ediff)
                Pkinetic = pTEST/PR

                u = random.random()
                if (Pkinetic < u):
                    if (debug): print("initQP diatom iteration.... REJECTED for (Kvib>0) (r,Ediff,Pkin) = ", (r,Ediff,Pkinetic))
                    continue

                if (debug): print("initQP diatom iteration.... ACCEPTED for (Kvib>0) (r,Ediff,Pkin) = ", (r,Ediff,Pkinetic))
                break

        # Determine whether to spin clockwise or anticlockwise
        u = random.random()
        if (u < 0.5e0):
            PR = -PR

        # Now, give the diatom momenta
        pBOTH = chemist_sys.momenta
        p = np.copy(pBOTH)
        p = np.zeros(p.shape)

        # First, give it vibrational momenta
        vrel = PR / mu
        vel1 = vrel * m[1] / (m[0] + m[1])
        vel2 = vel1 - vrel
        p[0][2] = m[0] * vel1
        p[1][2] = m[1] * vel2

        # Second, get a random rotation axis
        u = 2*np.pi * random.random()
        L = np.zeros((3))
        L[0] = AM*np.sin(u)
        L[1] = AM*np.cos(u)
        Ixy = mu * r * r

        # Third, give it rotational momenta
        q = np.zeros(np.shape(q))
        q[0][2] -= r * m[1] / (m[0] + m[1])
        q[1][2] += r * m[0] / (m[0] + m[1])
        omega = np.zeros((3))
        omega[0] = -L[0] / Ixy
        omega[1] = -L[1] / Ixy
        p += getMomentaFromAngularVelocity(m,q,omega)

        if (debug):
            print("initQP Evib and potEvib: ", Evib, Vdiff)
            print("initQP Erot and omega: ", getErotAndOmegaFromP(m,q,p))

        # Set these positions and momenta
        qBOTH = np.copy(q)
        chemist_sys.set_positions(qBOTH)

        pBOTH = np.copy(p)
        chemist_sys.set_momenta(pBOTH)

        sampled_momenta = PointSetD()
        for momentum in chemist_sys.momenta:
            sampled_momenta.push_back(PointD(*momentum))

        rv = self.results()
        return pt.wrap_results(rv, chemist_sys, sampled_momenta)

################################################################################################################

# Bimolecular sampling modules

class bimolecularSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(BimolecularSamplingPT())
        self.description("Do a bimolecular sampling")
        self.add_input("surface multiplicity")
        self.add_input("center of mass distance")
        self.add_input("collision energy")
        self.add_input("keywords").set_default({})
#       self.add_submodule(EnergyNuclearGradientStdVectorD(),'energy and gradient submodule')
#       self.add_submodule(HessianAndFrequencyPT(),'frequency submodule')
#       self.add_submodule(OptimizerPT(),'optimization submodule')

        self.add_submodule(ImpactParameterSamplingPT(),'impact parameter sampling submodule')
        self.add_submodule(UnimolecularSamplingPT(),'unimolecular sampling A submodule')
        self.add_submodule(UnimolecularSamplingPT(),'unimolecular sampling B submodule')

    def run_(self, inputs, submods):
        pt = BimolecularSamplingPT()
        molA0, molB0, = pt.unwrap_inputs(inputs)
        multAB = inputs['surface multiplicity'].value()
        dCM = inputs['center of mass distance'].value()
        collisionEnergy = inputs['collision energy'].value()
        keywords = inputs['keywords'].value()

        bMOD = submods["impact parameter sampling submodule"]
        samplingMODa = submods["unimolecular sampling A submodule"]
        samplingMODb = submods["unimolecular sampling B submodule"]

        kcalPERmol = 1.0

        debug = True

        b = bMOD.run_as(ImpactParameterSamplingPT())
        sampled_molA, sampled_pA = samplingMODa.run_as(UnimolecularSamplingPT(), molA0)
        sampled_molB, sampled_pB = samplingMODb.run_as(UnimolecularSamplingPT(), molB0)

        pA = []; pB = []
        for i in range(sampled_molA.molecule.size()):
            pi = sampled_pA.at(i)
            pA.append([pi.x,pi.y,pi.z])
        for i in range(sampled_molB.molecule.size()):
            pi = sampled_pB.at(i)
            pB.append([pi.x,pi.y,pi.z])

        sampled_sysA = dynamicChemicalSystem(sampled_molA,momenta=np.array(pA)); sampled_sysA.centerMolecule()
        sampled_sysB = dynamicChemicalSystem(sampled_molB,momenta=np.array(pB)); sampled_sysB.centerMolecule()

        massA = sum(sampled_sysA.masses)
        massB = sum(sampled_sysB.masses)
        totalMass = massA + massB

        # Construct the  vector between the two
        # molecules' centers of mass
        dPARALLEL = np.sqrt(dCM**2 - b**2)
        rPARALLEL = np.array([1.0, 0.0, 0.0])
        rREL = np.array([dPARALLEL, b, 0.0])

        # First, prepare the impact parameter with this vector
        # orthogonal to the centers of mass vector
        qB = np.copy(sampled_sysB.positions)
        for i in range(len(qB)):
            qB[i] += rREL
        sampled_sysB.set_positions(qB)

        # Second, prepare the center of mass velocities given the
        # collision energy
        reducedmass = massA*massB/(totalMass)
        speedREL = np.sqrt(2*(collisionEnergy*(kcalPERmol))/reducedmass)
        vAtrans = -(massB/totalMass)*(speedREL)*rPARALLEL
        vBtrans =  (massA/totalMass)*(speedREL)*rPARALLEL

        pAB = []
        chemist_mol = Molecule()

        vA = sampled_sysA.velocities
        pA = sampled_sysA.momenta
        for i in range(len(vA)):
            vA[i] += vAtrans
            pA[i] = vA[i] * sampled_sysA.masses[i]
            chemist_mol.push_back(Atom(sampled_sysA.atomnames[i],sampled_sysA.Zs[i],sampled_sysA.masses[i], *sampled_sysA.positions[i]))
            pAB.append(pA[i])

        vB = sampled_sysB.velocities
        pB = sampled_sysB.momenta
        for i in range(len(vB)):
            vB[i] += vBtrans
            pB[i] = vB[i] * sampled_sysB.masses[i]
            chemist_mol.push_back(Atom(sampled_sysB.atomnames[i],sampled_sysB.Zs[i],sampled_sysB.masses[i], *sampled_sysB.positions[i]))
            pAB.append(pB[i])

        chemist_mol.set_charge(sampled_sysA.charge + sampled_sysB.charge)
        chemist_mol.set_multiplicity(multAB)
        chemist_sys = ChemicalSystem(chemist_mol)

        sampled_momenta = PointSetD()
        for momentum in pAB:
            sampled_momenta.push_back(PointD(*momentum))

        rv = self.results()
        return pt.wrap_results(rv, chemist_sys, sampled_momenta)



################################################################################################################

# Sampling classe for the impact parameter

class fixedImpactParameterSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(ImpactParameterSamplingPT())
        self.description("Do a fixed impact parameter sampling (so just spit out the max value)")
        self.add_input("maximum impact parameter")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = ImpactParameterSamplingPT()
        bmax = inputs['maximum impact parameter'].value()
        keywords = inputs['keywords'].value()

        rv = self.results()
        return pt.wrap_results(rv, bmax)

class uniformImpactParameterSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(ImpactParameterSamplingPT())
        self.description("Do a uniform impact parameter sampling")
        self.add_input("maximum impact parameter")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = ImpactParameterSamplingPT()
        bmax = inputs['maximum impact parameter'].value()
        keywords = inputs['keywords'].value()

        u = random.random()
        b = u * bmax

        rv = self.results()
        return pt.wrap_results(rv, b)

class linearImpactParameterSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(ImpactParameterSamplingPT())
        self.description("Do a linear impact parameter sampling")
        self.add_input("maximum impact parameter")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = ImpactParameterSamplingPT()
        bmax = inputs['maximum impact parameter'].value()
        keywords = inputs['keywords'].value()

        while True:
            u = random.random()
            v = random.random()
            if (v < u): break

        b = u * bmax

        rv = self.results()
        return pt.wrap_results(rv, b)


################################################################################################################

# Sampling classes for rotation and vibration

class symmetricTopMoleculeAngularMomentumThermalSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(AngularMomentumThermalSamplingPT())
        self.description("Do an angular momentum sampling assuming the molecule is a symmetric top")
        self.add_input("rotational temperature")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = AngularMomentumThermalSamplingPT()
        axesMasses, = pt.unwrap_inputs(inputs)
        T = inputs['rotational temperature'].value()
        keywords = inputs['keywords'].value()

        self.kB=1.0

        Erot = 0.0e0
        L = np.zeros((3))
        axesMasses = [axesMasses.x, axesMasses.y, axesMasses.z]

        # Look at the difference between the "axesMasses" or the
        # moment of inertia principal components to see which
        # axes are favored to be "spun on" as Lz
        dI12 = axesMasses[0]-axesMasses[1]
        dI23 = axesMasses[1]-axesMasses[2]
        if (dI12 <= dI23):
            zAxis=2
        else:
            zAxis=0

        # Define a maximum L (idk why this value in particular)
        LzMAX = np.sqrt(20.0e0*axesMasses[zAxis]*self.kB*T)

        # Do rejection sampling to determine Lz
        while True:
            u = random.random()
            L[zAxis] = u * LzMAX
            probL = np.exp(-(L[zAxis]**2)/(2*axesMasses[zAxis]*self.kB*T))
            u = random.random()
            if (u <= probL): break

        # Finally, flip a coin to determine the sign of Lz
        u = random.random()
        if (u > 0.5e0): L[zAxis] = -L[zAxis]

        # Calculate the Erot contribution from this
        Erot = (L[zAxis]**2)/axesMasses[zAxis]

        # Here, the z-axis is the third spatial dimension
        if (zAxis==2):

            # Use the inverse CDF to determine Lxy
            u = random.random()
            Ixy = np.sqrt(axesMasses[0]*axesMasses[1])
            Lxyz = np.sqrt(L[2]**2 - 2*Ixy*self.kB*T*np.log(1.0e0-u))
            Lxy = np.sqrt(Lxyz**2 - L[2]**2)

            # Determine a random phase for the x-y partitioning
            # of the Lxy component of the angular momentum
            u = 2 * np.pi * random.random()
            L[0] = Lxy*np.sin(u)
            L[1] = Lxy*np.cos(u)

            # Calculate the Erot contribution from this
            Erot = 0.5e0*(Erot + ((L[0]**2)/axesMasses[0]) + ((L[1]**2)/axesMasses[1]))

        # Here, the z-axis is the first spatial dimension
        else:

            # Use the inverse CDF to determine Lxy
            u = random.random()
            Ixy = np.sqrt(axesMasses[1]*axesMasses[2])
            Lxyz = np.sqrt(L[0]**2 - 2*Ixy*self.kB*T*np.log(1.0e0-u))
            Lxy = np.sqrt(Lxyz**2 - L[0]**2)

            # Determine a random phase for the x-y partitioning
            # of the Lxy component of the angular momentum
            u = 2 * np.pi * random.random()
            L[1] = Lxy*np.sin(u)
            L[2] = Lxy*np.cos(u)

            # Calculate the Erot contribution from this
            Erot = 0.5e0*(Erot + ((L[1]**2)/axesMasses[1]) + ((L[2]**2)/axesMasses[2]))

        rv = self.results()
        return pt.wrap_results(rv, Erot, PointD(*L))

class linearMoleculeAngularMomentumThermalSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(AngularMomentumThermalSamplingPT())
        self.description("Do an angular momentum sampling assuming the molecule is linear")
        self.add_input("rotational temperature")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = AngularMomentumThermalSamplingPT()
        axesMasses, = pt.unwrap_inputs(inputs)
        T = inputs['rotational temperature'].value()
        keywords = inputs['keywords'].value()

        self.kB=1.0

        Erot = 0.0e0
        L = np.zeros((3))
        axesMasses = [axesMasses.x, axesMasses.y, axesMasses.z]  # Third axis should be zero/undefined

        # Use the inverse CDF to determine Lxy
        u = random.random()
        Ixy = np.sqrt(axesMasses[0]*axesMasses[1])
        Lxyz = np.sqrt(L[2]**2 - 2*Ixy*self.kB*T*np.log(1.0e0-u))
        Lxy = np.sqrt(Lxyz**2 - L[2]**2)

        # Determine a random phase for the x-y partitioning
        # of the Lxy component of the angular momentum
        u = 2 * np.pi * random.random()
        L[0] = Lxy*np.sin(u)
        L[1] = Lxy*np.cos(u)

        # Calculate the Erot contribution from this
        Erot = 0.5e0*(Erot + ((L[0]**2)/axesMasses[0]) + ((L[1]**2)/axesMasses[1]))

        rv = self.results()
        return pt.wrap_results(rv, Erot, PointD(*L))

class vibrationalQuantaThermalSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(VibrationalQuantaThermalSamplingPT())
        self.description("Do a vibrational quanta sampling from a thermal distribution")
        self.add_input("vibrational temperature")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = VibrationalQuantaThermalSamplingPT()
        freq, = pt.unwrap_inputs(inputs)
        T = inputs['vibrational temperature'].value()
        keywords = inputs['keywords'].value()

        self.kB=1.0

        u = random.random()
        n = int(-np.log(u)*T*self.kB/freq)

        rv = self.results()
        return pt.wrap_results(rv, n)


################################################################################################################


def main():
    n_threads = 6
    mpiexec_command = "srun --overlap --mpi=pmix --nodes={nnodes} --ntasks-per-node={ranks_per_node} --ntasks={total_ranks} --cpus-per-task={cores_per_rank}" # For QCEngine, it has to be of this format: (1) nodes, (2) ranks_per_node, (3) total_ranks, (4) cores_per_rank
    MPIconfig = {
        "use_mpiexec" : True,
        "mpiexec_command": mpiexec_command,
        "nnodes": 1,
        "ncores": n_threads,
        "cores_per_rank": 1,
    }

    chargeA = 0; multA = 1
    samplingA = "thermal"; rSampleA = 10.0; vSampleA = 10.0

    chargeB = 0; multB = 1
    samplingB = "thermal"; rSampleB = 298.15; vSampleB = 298.15
#   samplingB = "semiclassical"; rSampleB = 3; vSampleB = 0

    multAB = 1

    bmax = 1.0
    rCM = 10.0
    collisionEnergy = 5.0
    samplingIP = "fixed"

    ###################################################################################

    mm = pp.ModuleManager()

    # Load the PES

    abinitio_key="NWChem : SCF Gradient"
    mm.add_module(abinitio_key, NWChemEnergyAndGradientViaMolSSI())
    mm.change_input(abinitio_key, 'method', 'SCF')
    mm.change_input(abinitio_key, 'basis set', 'sto-3g')
#   mm.change_input(abinitio_key, 'method', 'B3LYP')
#   mm.change_input(abinitio_key, 'basis set', '6-31G*')
    mm.change_input(abinitio_key, 'MPI config', MPIconfig)

    abinitio_justgrad_key="NWChem : SCF Only Gradient"
    mm.add_module(abinitio_justgrad_key, NWChemGradientViaMolSSI())
    mm.change_input(abinitio_justgrad_key, 'method', 'SCF')
    mm.change_input(abinitio_justgrad_key, 'basis set', 'sto-3g')
#   mm.change_input(abinitio_justgrad_key, 'method', 'B3LYP')
#   mm.change_input(abinitio_justgrad_key, 'basis set', '6-31G*')
    mm.change_input(abinitio_justgrad_key, 'MPI config', MPIconfig)

    ###################################################################################

    # Load the helper functions

    module_key="VENUS : frequency"
    mm.add_module(module_key, numericalHessianAndFrequencyAnalysis())
#   mm.change_submod(module_key, 'energy and gradient submodule', abinitio_key)
    mm.change_submod(module_key, 'gradient submodule', abinitio_justgrad_key)
    mm.change_input(module_key, 'numerical derivative geometry step size', 0.001)

    module_key="VENUS : simple optimization"
    mm.add_module(module_key, simpleGradientDescentLineSearch())
    mm.change_submod(module_key, 'energy and gradient submodule', abinitio_key)
    mm.change_input(module_key, 'default step size', 1.0) 
    mm.change_input(module_key, 'gradient convergence threshold', 5.0e-4)

    ###################################################################################

    # Load the sampling modules

    if (samplingA == "thermal"):
        module_key="VENUS : nonlinear molecule angular momentum thermal sampling A"
        mm.add_module(module_key, symmetricTopMoleculeAngularMomentumThermalSampling())
        mm.change_input(module_key, 'rotational temperature', rSampleA)
        mm.at(module_key).turn_off_memoization()
    
        module_key="VENUS : linear molecule angular momentum thermal sampling A"
        mm.add_module(module_key, linearMoleculeAngularMomentumThermalSampling())
        mm.change_input(module_key, 'rotational temperature', rSampleA)
        mm.at(module_key).turn_off_memoization()
    
        module_key="VENUS : vibrational quanta thermal sampling A"
        mm.add_module(module_key, vibrationalQuantaThermalSampling())
        mm.change_input(module_key, 'vibrational temperature', vSampleA)
        mm.at(module_key).turn_off_memoization()

        module_key="VENUS : unimolecular sampling A"
        mm.add_module(module_key, unimolecularThermalSampling())
        mm.change_submod(module_key, 'energy and gradient submodule', abinitio_key)
        mm.change_submod(module_key, 'frequency submodule', "VENUS : frequency")
        mm.change_submod(module_key, 'optimization submodule', "VENUS : simple optimization")
        mm.change_submod(module_key, 'nonlinear rotation sampling submodule', "VENUS : nonlinear molecule angular momentum thermal sampling A")
        mm.change_submod(module_key, 'linear rotation sampling submodule', "VENUS : linear molecule angular momentum thermal sampling A")
        mm.change_submod(module_key, 'vibration sampling submodule', "VENUS : vibrational quanta thermal sampling A")
        mm.change_input(module_key, 'rotational sample', rSampleA)
        mm.change_input(module_key, 'vibrational sample', vSampleA)
        mm.at(module_key).turn_off_memoization()

    elif (samplingA == "semiclassical"):
        module_key="VENUS : unimolecular sampling A"
        mm.add_module(module_key, unimolecularSemiclassicalSampling())
        mm.change_submod(module_key, 'energy and gradient submodule', abinitio_key)
        mm.change_submod(module_key, 'frequency submodule', "VENUS : frequency")
        mm.change_submod(module_key, 'optimization submodule', "VENUS : simple optimization")
        mm.change_input(module_key, 'rotational sample', rSampleA)
        mm.change_input(module_key, 'vibrational sample', vSampleA)
        mm.at(module_key).turn_off_memoization()

    else:
        raise ValueError("In unimolecular sampling A .... only 'thermal' or 'semiclassical' (for diatoms) sampling is allowed for now")


    if (samplingB == "thermal"):
        module_key="VENUS : nonlinear molecule angular momentum thermal sampling B"
        mm.add_module(module_key, symmetricTopMoleculeAngularMomentumThermalSampling())
        mm.change_input(module_key, 'rotational temperature', rSampleB)
        mm.at(module_key).turn_off_memoization()
    
        module_key="VENUS : linear molecule angular momentum thermal sampling B"
        mm.add_module(module_key, linearMoleculeAngularMomentumThermalSampling())
        mm.change_input(module_key, 'rotational temperature', rSampleB)
        mm.at(module_key).turn_off_memoization()
    
        module_key="VENUS : vibrational quanta thermal sampling B"
        mm.add_module(module_key, vibrationalQuantaThermalSampling())
        mm.change_input(module_key, 'vibrational temperature', vSampleB)
        mm.at(module_key).turn_off_memoization()

        module_key="VENUS : unimolecular sampling B"
        mm.add_module(module_key, unimolecularThermalSampling())
        mm.change_submod(module_key, 'energy and gradient submodule', abinitio_key)
        mm.change_submod(module_key, 'frequency submodule', "VENUS : frequency")
        mm.change_submod(module_key, 'optimization submodule', "VENUS : simple optimization")
        mm.change_submod(module_key, 'nonlinear rotation sampling submodule', "VENUS : nonlinear molecule angular momentum thermal sampling B")
        mm.change_submod(module_key, 'linear rotation sampling submodule', "VENUS : linear molecule angular momentum thermal sampling B")
        mm.change_submod(module_key, 'vibration sampling submodule', "VENUS : vibrational quanta thermal sampling B")
        mm.change_input(module_key, 'rotational sample', rSampleB)
        mm.change_input(module_key, 'vibrational sample', vSampleB)
        mm.at(module_key).turn_off_memoization()

    elif (samplingB == "semiclassical"):
        module_key="VENUS : unimolecular sampling B"
        mm.add_module(module_key, unimolecularSemiclassicalSampling())
        mm.change_submod(module_key, 'energy and gradient submodule', abinitio_key)
        mm.change_submod(module_key, 'frequency submodule', "VENUS : frequency")
        mm.change_submod(module_key, 'optimization submodule', "VENUS : simple optimization")
        mm.change_input(module_key, 'rotational sample', rSampleB)
        mm.change_input(module_key, 'vibrational sample', vSampleB)
        mm.at(module_key).turn_off_memoization()

    else:
        raise ValueError("In unimolecular sampling B .... only 'thermal' or 'semiclassical' (for diatoms) sampling is allowed for now")


    if (samplingIP == "fixed"):
        module_key="VENUS : impact parameter sampling"
        mm.add_module(module_key, fixedImpactParameterSampling())
        mm.change_input(module_key, 'maximum impact parameter', bmax)
        mm.at(module_key).turn_off_memoization()

    elif (samplingIP == "uniform"):
        module_key="VENUS : impact parameter sampling"
        mm.add_module(module_key, uniformImpactParameterSampling())
        mm.change_input(module_key, 'maximum impact parameter', bmax)
        mm.at(module_key).turn_off_memoization()

    elif (samplingIP == "linear"):
        module_key="VENUS : impact parameter sampling"
        mm.add_module(module_key, linearImpactParameterSampling())
        mm.change_input(module_key, 'maximum impact parameter', bmax)
        mm.at(module_key).turn_off_memoization()

    else:
        raise ValueError("In the impact parameter sampling, only 'fixed', 'uniform', or 'linear' is allowed for now")

    ###################################################################################

    module_key="VENUS : bimolecular sampling"
    mm.add_module(module_key, bimolecularSampling())
#   mm.change_submod(module_key, 'energy and gradient submodule', abinitio_key)
#   mm.change_submod(module_key, 'frequency submodule', "VENUS : frequency")
#   mm.change_submod(module_key, 'optimization submodule', "VENUS : simple optimization")
    mm.change_submod(module_key, 'unimolecular sampling A submodule', "VENUS : unimolecular sampling A")
    mm.change_submod(module_key, 'unimolecular sampling B submodule', "VENUS : unimolecular sampling B")
    mm.change_submod(module_key, 'impact parameter sampling submodule', "VENUS : impact parameter sampling")
    mm.change_input(module_key, 'surface multiplicity', multAB)
    mm.change_input(module_key, 'center of mass distance', rCM)
    mm.change_input(module_key, 'collision energy', collisionEnergy)
    mm.at(module_key).turn_off_memoization()

    debug = True

    # Molecule A
    chemist_mol = Molecule()
#   chemist_mol.push_back(Atom("H",1, 1.008,  0.96741185,  0.43340757, 0.0))
#   chemist_mol.push_back(Atom("H",1, 1.008,  0.96641185, -1.43340757, 0.0))

#   chemist_mol.push_back(Atom("H",1, 1.008,  1.01913080,  1.62835915, 0.0))
#   chemist_mol.push_back(Atom("C",6,12.011, -0.33971027,  0.00000000, 0.0))
#   chemist_mol.push_back(Atom("H",1, 1.008,  1.01913080, -1.62835915, 0.0))

    chemist_mol.push_back(Atom("H",1, 1.008,  0.94371749,  1.43950432,0.0))
    chemist_mol.push_back(Atom("O",8,15.999, -0.18706553,  0.00021411, 0.0))
    chemist_mol.push_back(Atom("H",1, 1.008,  0.94289937, -1.43971843, 0.0))

#   chemist_mol.push_back(Atom("H",1, 1.008,  1.01258959,  1.53589180, -0.05198102))
#   chemist_mol.push_back(Atom("N",7,14.007, -0.14798652,  0.00011056, -0.09203935))
#   chemist_mol.push_back(Atom("H",1, 1.008, -1.07593631,  0.00020490,  1.59620733))
#   chemist_mol.push_back(Atom("H",1, 1.008,  1.01188456, -1.53620726, -0.05218695))

#   chemist_mol.push_back(Atom("C",6,12.011, -0.05483732,  0.00004186, -0.03920710))
#   chemist_mol.push_back(Atom("H",1, 1.008,  1.13656056,  1.68719968, -0.05234013))
#   chemist_mol.push_back(Atom("H",1, 1.008, -1.22740897,  0.00022398,  1.66288609))
#   chemist_mol.push_back(Atom("H",1, 1.008, -1.26457760,  0.00033591, -1.71509339))
#   chemist_mol.push_back(Atom("H",1, 1.008,  1.13587834, -1.68759653, -0.05245278))

    chemist_mol.set_charge(chargeA)
    chemist_mol.set_multiplicity(multA)
    chemist_sys_A = ChemicalSystem(chemist_mol)

    # Molecule B
    chemist_mol = Molecule()
    chemist_mol.push_back(Atom("C",6,12.011,  0.96583093, -1.57486194, 0.0))
    chemist_mol.push_back(Atom("O",8,15.999,  0.96799277,  0.57486194, 0.0))
    chemist_mol.set_charge(chargeB)
    chemist_mol.set_multiplicity(multB)
    chemist_sys_B = ChemicalSystem(chemist_mol)

    Nsamples = 3
    for i in range(Nsamples):
        module_key = "VENUS : nonlinear molecule angular momentum thermal sampling A"
        sampled_Erot, sampled_l = mm.run_as(AngularMomentumThermalSamplingPT(), module_key, PointD(1.0,1.0,1.0))
        module_key="VENUS : vibrational quanta thermal sampling A"
        sampled_n = mm.run_as(VibrationalQuantaThermalSamplingPT(), module_key, 100.0)

        if (debug):
            print("AM from thermal:", np.sqrt(sampled_l.x**2 + sampled_l.y**2 + sampled_l.z**2))
            print("we got here: ", mm.at(module_key).profile_info())

        ########################################################################################################

        module_key = "VENUS : bimolecular sampling"
        sampled_chemist_sys, sampled_p = mm.run_as(BimolecularSamplingPT(), module_key, chemist_sys_A, chemist_sys_B)

        Natoms = sampled_chemist_sys.molecule.size()
        print(Natoms)
        print("WHAT: sampled_l:", sampled_l.x, sampled_l.y, sampled_l.z, " Erot: ", sampled_Erot, " sampled_n:", sampled_n)
        for i, atom in enumerate(sampled_chemist_sys.molecule):
            pi = sampled_p.at(i)
            print("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}    {4:18.10f} {5:18.10f} {6:18.10f}".format(atom.name, atom.x, atom.y, atom.z, pi.x, pi.y, pi.z))
        print("")

        ########################################################################################################

##      module_key = "VENUS : unimolecular sampling B"
##      sampled_chemist_sys_B, sampled_p_B = mm.run_as(UnimolecularSamplingPT(), module_key, chemist_sys_B)

##      Natoms = sampled_chemist_sys_B.molecule.size()
##      print(Natoms)
##      print("sampled_l:", sampled_l.x, sampled_l.y, sampled_l.z, " Erot: ", sampled_Erot, " sampled_n:", sampled_n)
##      for i, atom in enumerate(sampled_chemist_sys_B.molecule):
##          pi = sampled_p_B.at(i)
##          print("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}    {4:18.10f} {5:18.10f} {6:18.10f}".format(atom.name, atom.x, atom.y, atom.z, pi.x, pi.y, pi.z))
##      print("")

##      ########################################################################################################

##      module_key = "VENUS : unimolecular sampling A"
##      sampled_chemist_sys_A, sampled_p_A = mm.run_as(UnimolecularSamplingPT(), module_key, chemist_sys_A)

##      Natoms = sampled_chemist_sys_A.molecule.size()
##      print(Natoms)
##      print("sampled_l:", sampled_l.x, sampled_l.y, sampled_l.z, " Erot: ", sampled_Erot, " sampled_n:", sampled_n)
##      for i, atom in enumerate(sampled_chemist_sys_A.molecule):
##          pi = sampled_p_A.at(i)
##          print("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}    {4:18.10f} {5:18.10f} {6:18.10f}".format(atom.name, atom.x, atom.y, atom.z, pi.x, pi.y, pi.z))
##      print("")

        ########################################################################################################

print("BEFORE main()")
main()
print(" AFTER main()")
