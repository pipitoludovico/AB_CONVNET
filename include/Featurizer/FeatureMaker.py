import os.path
from os import cpu_count, chdir, makedirs
from multiprocessing import Pool
from subprocess import run
import numpy as np
import logging


class FeaturizerClass:
    def __init__(self, dbDict: dict, root: str):
        self.dbDict = dbDict
        self.root = root
        self.datFile = "FINAL_DECOMP_MMPBSA.dat"

    def Featurize(self, pdbID: str) -> None:
        if not os.path.exists("selected/" + pdbID + "/" + self.datFile):
            chdir(self.root)
            return
        try:
            chdir("selected/" + pdbID)
            dec_res = self.GetDecomp()
            self.BuildMatrix(pdbID, dec_res)
        except Exception as e:
            logging.error(f"Error processing {pdbID}: {e}")
        finally:
            chdir(self.root)

    def GetDecomp(self) -> (list, list):
        dec_results = {}

        def WriteTotalFromDecomp():
            decomp_purged = open("total_purged.csv", 'w')
            with open(f'{self.datFile}', 'r') as f:
                for purgedLine in f.readlines():
                    decomp_purged.writelines(purgedLine)
                    if purgedLine == "\n" or purgedLine == " ":
                        break

        def GetResults():
            with open('total_purged.csv', 'r') as f:
                after_header = f.readlines()[7:]
                for lines in after_header:
                    if lines == "" or lines == " " or lines == "\n":
                        continue
                    if len(lines.split()) > 3:
                        if lines.split()[1].split(',')[1] == "R" or lines.split()[1].split(',')[1] == 'L':
                            resname = lines[0:4].strip()
                            resnum = lines[4:7].strip()
                            total_energy = float(lines.split(",")[-3])
                            if (resname + resnum) not in dec_results:
                                dec_results[resname + resnum] = total_energy

        WriteTotalFromDecomp()
        GetResults()
        return dec_results

    @staticmethod
    def BuildMatrix(pdbID: str, dec_res_: dict) -> np.array:
        elements = {'H': 1.008, 'He': 4.002, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011, 'N': 14.007,
                    'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085,
                    'P': 30.974, 'S': 32.066, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956,
                    'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
                    'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.64, 'As': 74.922, 'Se': 78.96, 'Br': 79.904,
                    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96,
                    'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818,
                    'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.904, 'Xe': 131.293, 'Cs': 132.905, 'Ba': 137.327,
                    'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242, 'Pm': 145.0, 'Sm': 150.36,
                    'Eu': 151.964,
                    'Gd': 157.25, 'Tb': 158.925, 'Dy': 162.5, 'Ho': 164.930, 'Er': 167.259, 'Tm': 168.934, 'Yb': 173.04,
                    'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.948, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                    'Pt': 195.084, 'Au': 196.967, 'Hg': 200.59, 'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.980,
                    'Th': 232.038,
                    'Pa': 231.036, 'U': 238.029}
        finalCoordinates = {}
        tempPesto = {}
        GBSA: float = -1.0

        # we get the total GBSA
        with open('results_mmgbsa.dat', 'r') as GBSAfile:
            for line in GBSAfile.readlines():
                if "DELTA TOTAL" in line:
                    GBSA = float(line.split()[2])
                    break
        # we convert the complex to get the information we need
        if not os.path.exists('complex_minimized.mol2'):
            run("obabel -i pdb complex_minimized_chains.pdb -o mol2 -O complex_minimized.mol2 > logs/obabelLog.log",
                shell=True)
        couples = []
        # we check contact points between Ab and Ag and make the pairs
        with open('contacts.int', 'r') as contacts:
            contact_content = contacts.read()
            results = contact_content.replace("\n", '').split(',')[1:]
            # numContacts = contact_content.split(",")[0]
            for couple in results:
                couples.append(couple)
        # and we add, if any, the % of molecular surface patch % to the score. This is where we build the final matrix.
        with open('complex_minimized.mol2', 'r') as complexMOL2, open('patches/selection_i0.pdb') as PestoFile:
            finalCoordinates[pdbID] = {}
            tempPesto[pdbID] = {}
            pestoPDB = PestoFile.readlines()
            for line in pestoPDB:
                if len(line.split()) > 5:
                    residueID = line.split()[3] + line[23:27].strip()
                    interfaceProb = line[56:62].strip()
                    tempPesto[pdbID][str(residueID)] = interfaceProb

            for line in complexMOL2.readlines():
                if len(line.split()) > 5:
                    try:
                        x, y, z, atomType, partialCharge, ResidAndResnum = line.split()[2], line.split()[3], line.split()[4], line.split()[5], line.split()[8], line.split()[7]
                        x, y, z, partialCharge = map(float, [x, y, z, partialCharge])
                        prob = float(tempPesto[pdbID].get(ResidAndResnum, 0))
                        if ResidAndResnum not in finalCoordinates[pdbID]:
                            finalCoordinates[pdbID][ResidAndResnum] = []
                            finalCoordinates[pdbID][ResidAndResnum].append(
                                np.array([elements[atomType.split(".")[0]], partialCharge, x, y, z, prob]))
                        else:
                            finalCoordinates[pdbID][ResidAndResnum].append(
                                np.array([elements[atomType.split(".")[0]], partialCharge, x, y, z, prob]))
                    except IndexError:
                        print(pdbID, "HAD AN INDEX OUT OF RANGE IN THE MOL2 FILE.")

        dataList = []
        for pair in couples:
            res1, res2, = pair.split("-")[0], pair.split("-")[1]
            dec1, dec2 = dec_res_.get(res1, 0), dec_res_.get(res2, 0)
            combDecomp = float(float(dec1) + float(dec2))
            arr1, arr2 = np.array(finalCoordinates[pdbID][res1]), np.array(finalCoordinates[pdbID][res2])
            arrayList = [arr1, arr2]
            max_len = max([arr.shape[0] for arr in arrayList])
            # we pad before stacking
            if arr1.shape[0] < arr2.shape[0]:
                paddedarr1 = np.pad(arr1, ((0, max_len - arr1.shape[0]), (0, 0)), 'constant', constant_values=0)
                paddedarr2 = arr2
            else:
                paddedarr1 = arr1
                paddedarr2 = np.pad(arr2, ((0, max_len - arr2.shape[0]), (0, 0)), 'constant', constant_values=0)
            a3 = np.hstack((paddedarr1, paddedarr2))
            combDecomp_array = np.full((a3.shape[0], 1), combDecomp)
            GBSA_array = np.full((a3.shape[0], 1), GBSA)
            a4 = np.hstack((a3, combDecomp_array, GBSA_array))
            zero_count = np.sum(a4 == 0, axis=1)
            filtered_arr = a4[zero_count < 6]
            dataList.append(filtered_arr)
        dataMatrix = np.vstack(dataList)
        makedirs('saved_results', exist_ok=True)
        with open('../../summary', 'a') as summaryLog:
            summaryLog.write(f"\nTotal number of pair contacts in pdb {pdbID} = {dataMatrix.shape}")
        if os.path.exists('./saved_results/protein_data.npy'):
            with open('./saved_results/protein_data.npy', 'wb') as f:
                np.save(f, dataMatrix, allow_pickle=True)
        else:
            np.save("./saved_results/protein_data.npy", dataMatrix, allow_pickle=True)

    def ParallelFeaturize(self) -> None:
        cpuUnits = int(cpu_count() // 4)
        with Pool(processes=cpuUnits) as p:
            for pdbID, chains in self.dbDict.items():
                p.apply_async(self.Featurize, args=(pdbID,))
            p.close()
            p.join()
        logging.debug("All processes have completed.")
