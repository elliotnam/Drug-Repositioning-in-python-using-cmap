import openbabel
import pandas as pd
import numpy as np


obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("smi", "mdl")

mol = openbabel.OBMol()
obConversion.ReadString(mol, "CN(C)C(=N)N=C(N)N")

print 'Should print 5 (atoms)'
print mol.NumAtoms()

mol.AddHydrogens()
print 'Should print 9 (atoms) after adding hydrogens'
print mol.NumAtoms()

outMDL = obConversion.WriteString(mol)
print(outMDL)

text_file = open("/home/elliotnam/project/cmap/tt.txt","w")
text_file.write(outMDL)
text_file.close()