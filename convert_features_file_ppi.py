import numpy as np
import json
features = np.load("C:/Users/danie/Desktop/ITA/CES27_distrProg/labExame/ppi/ppi-feats.npy")
with open("C:/Users/danie/Desktop/ITA/CES27_distrProg/labExame/ppi/ppi-id_map.json", "r") as f:
    id_map = json.load(f)

with open("output_convert_features_file_ppi.txt", "w") as f:
    for k, v in id_map.items():
        f.write("{};[{}]\n".format(k, ",".join(str(x) for x in features[v])))
