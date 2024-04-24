import pluginplay as pp
from chemist import Atom, Molecule, ChemicalSystem, PointSetD
from venus_nwchemex_plugin import UnimolecularSamplingPT
#from simde import TotalEnergy


class NWChemViaMolSSI(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(TotalEnergy())
        self.description("Calls NWChem via MolSSI's QCEngine")
        self.add_input('method')
        self.add_input("basis set")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = TotalEnergy()
        mol, = pt.unwrap_inputs(inputs)
        method = inputs['method'].value()
        basis = inputs['basis set'].value()
        keywords = inputs['keywords'].value()

        model = {"method": method, "basis": basis}
#       e = call_qcengine(pt, mol, 'nwchem', model=model, keywords=keywords)
        e = 0.0
        rv = self.results()
        return pt.wrap_results(rv, e)


class VENUSunimolecularSampling(pp.ModuleBase):

    def __init__(self):
        pp.ModuleBase.__init__(self)
        self.satisfies_property_type(UnimolecularSamplingPT())
        self.description("Do a unimolecular sampling")
        self.add_input('method')
        self.add_input("basis set")
        self.add_input("keywords").set_default({})

    def run_(self, inputs, submods):
        pt = UnimolecularSamplingPT()
        mol, = pt.unwrap_inputs(inputs)
        method = inputs['method'].value()
        basis = inputs['basis set'].value()
        keywords = inputs['keywords'].value()

        model = {"method": method, "basis": basis}
#       e = call_qcengine(pt, mol, 'nwchem', model=model, keywords=keywords)
        p = [0.0, 0.0, 0.0]
        rv = self.results()
        return pt.wrap_results(rv, mol, p)

def main():
    mm = pp.ModuleManager()
#   print("simde stuff:", dir(simde))
#   print("simde stuff:", vars(simde))
#   print("chemist stuff:", dir(chemist))
#   print("chemist stuff:", vars(chemist))
#   mm.add_module("Energy thing", NWChemViaMolSSI())
    mm.add_module("InitQP : unimolecular sampling", VENUSunimolecularSampling())


    chemist_mol = Molecule()
#   chemist_mol.push_back(Atom(anAtom.symbol,anAtom.number,anAtom.mass,*anAtom.position * self.Ang_to_R))
    chemist_mol.push_back(Atom("H",1,1.008,  1.2, 1.0,0.0))
    chemist_mol.push_back(Atom("C",6,12.011, 0.0, 0.0,0.0))
    chemist_mol.push_back(Atom("H",1,1.008,  1.2,-1.0,0.0))
    chemist_mol.set_charge(0)
    chemist_mol.set_multiplicity(1)
    chemist_sys = ChemicalSystem(chemist_mol)

    mm.change_input("InitQP : unimolecular sampling", 'method', "some method")
    mm.change_input("InitQP : unimolecular sampling", 'basis set', "some basis set")
    initQP = mm.run_as(VENUSunimolecularSampling(), "InitQP : unimolecular sampling", chemist_sys)

    print("initQP:", initQP)



print("BEFORE main()")
main()
print(" AFTER main()")
