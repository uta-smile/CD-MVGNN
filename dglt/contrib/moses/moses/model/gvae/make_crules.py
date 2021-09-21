import os

root_dir = '/data1/jonathan/Molecule.Generation/AIPharmacist'
rls_path = os.path.join(root_dir, 'data/crules.txt')

chars = ['(', ')', '[', ']', '-', '=', '#']
chars += [chr(ord('0') + x) for x in range(10)]
chars += [chr(ord('A') + x) for x in range(26)]
chars += [chr(ord('a') + x) for x in range(26)]

with open(rls_path, 'w') as o_file:
    o_file.write('\n'.join(chars))
