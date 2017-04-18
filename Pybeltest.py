import pybel2


smiles = ['CCCC', 'CCCN']
mols = [pybel2.readstring("smi", x) for x in smiles] # Create a list of two molecules
fps = [x.calcfp("ECFP6") for x in mols] # Calculate their fingerprints
print fps[0].bits, fps[1].bits
print fps[0] | fps[1] # Print the Tanimoto coefficient

