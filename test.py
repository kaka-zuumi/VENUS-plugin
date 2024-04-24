import pluginplay as pp
from chemist import Atom, Molecule, ChemicalSystem, PointD, PointSetD
from venus_nwchemex_plugin import UnimolecularSamplingPT
#from simde import TotalEnergy


class unimolecularThermalSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(UnimolecularSamplingPT())
        self.description("Do a unimolecular sampling")
        self.add_input("rotational sample")
        self.add_input("vibrational sample")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = UnimolecularSamplingPT()
        mol, = pt.unwrap_inputs(inputs)
        rSample = inputs['rotational sample'].value()
        vSample = inputs['vibrational sample'].value()
        keywords = inputs['keywords'].value()

        model = {"rotational sample": rSample, "vibrational sample": vSample}
#       e = call_qcengine(pt, mol, 'nwchem', model=model, keywords=keywords)
#       p = PointSetD([0.0, 0.0, 0.0])
        p = PointSetD()
        x = PointD(6.0,8.0,9.0)
        p.push_back(x)
        p.push_back(x)
        p.push_back(x)
        rv = self.results()
        return pt.wrap_results(rv, mol, p)

def main():
    mm = pp.ModuleManager()
    mm.add_module("VENUS : unimolecular thermal sampling", unimolecularThermalSampling())


    chemist_mol = Molecule()
#   chemist_mol.push_back(Atom(anAtom.symbol,anAtom.number,anAtom.mass,*anAtom.position * self.Ang_to_R))
    chemist_mol.push_back(Atom("H",1,1.008,  1.2, 1.0, 0.0))
    chemist_mol.push_back(Atom("C",6,12.011, 0.0, 0.0, 0.0))
    chemist_mol.push_back(Atom("H",1,1.008,  1.2,-1.0, 0.0))
    chemist_mol.set_charge(0)
    chemist_mol.set_multiplicity(1)
    chemist_sys = ChemicalSystem(chemist_mol)

    chemist_mol1 = Molecule()
    chemist_mol1.push_back(Atom("H",1,1.008,  1.2, 1.0, 0.0))
    chemist_mol1.push_back(Atom("H",1,1.008,  1.2,-1.0, 0.0))
    chemist_mol1.push_back(Atom("H",1,1.008,  6.2,-1.0, 0.0))
    chemist_mol1.push_back(Atom("H",1,1.008,  8.2,-1.0, 0.0))
    chemist_mol1.set_charge(0)
    chemist_mol1.set_multiplicity(1)
    chemist_sys1 = ChemicalSystem(chemist_mol1)

    Natoms = chemist_sys1.molecule.size()
    print(Natoms)
    print(dir(chemist_sys1))
    print(dir(chemist_sys1.molecule))
    print(type(chemist_sys1.molecule.at(3)))
    print(type(chemist_sys1.molecule.at(2)))
    print(type(chemist_sys1.molecule.at(1)))
    print(type(chemist_sys1.molecule.at(0)))
    a4 = chemist_sys1.molecule.at(3)
    a3 = chemist_sys1.molecule.at(2)
    a2 = chemist_sys1.molecule.at(1)
    a1 = chemist_sys1.molecule.at(0)
    print("GOT HERE")
    print(a4)
    print(a3)
    print(a2)
    print(dir(a1))
    print(a1.name(), a1.x, a1.y, a1.z)
    print(chemist_sys1.molecule.at(3))
    print(chemist_sys1.molecule.at(2))
    print(chemist_sys1.molecule.at(2))
    print(chemist_sys1.molecule.at(1))
    print(chemist_sys1.molecule.at(0))



    Natoms = chemist_sys.molecule.size()
    print(Natoms)
    print(dir(chemist_sys))
    print(dir(chemist_sys.molecule))
    print(chemist_sys.molecule.at(2))
    print(chemist_sys.molecule.at(1))
    print(chemist_sys.molecule.at(0))

    module_key = "VENUS : unimolecular thermal sampling"
    mm.change_input(module_key, 'rotational sample', 10.0)
    mm.change_input(module_key, 'vibrational sample', 298.15)
    sampled_chemist_sys, sampled_p = mm.run_as(UnimolecularSamplingPT(), module_key, chemist_sys)

#    Natoms = chemist_sys.molecule.size()
#    print(Natoms)
#    print(dir(chemist_sys.molecule))
#    print(chemist_sys.molecule.at(0))
#    print("initQP:", sampled_chemist_sys, sampled_p)
##    for i in range(Natoms):
##        print(chemist_sys.molecule.at(i), sampled_p.at(i))
###       print(sampled_chemist_sys.molecule.at(i), sampled_p.at(i))
##    print("P:", dir(sampled_p))
##    print("P:", sampled_p)
##    print("P:", sampled_p.at(0))
##    print("P:", dir(sampled_p.at(0)))
##    print("P:", sampled_p.at(0).x, sampled_p.at(0).y, sampled_p.at(0).z)



print("BEFORE main()")
main()
print(" AFTER main()")
