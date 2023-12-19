import numpy as np
from needle_shape_sensing.sensorized_needles import FBGNeedle

SRC_DIR = "/root/colcon_ws/src/needle_insertion_experiment_setup"

in_params = f"{SRC_DIR}/needle_shape_publisher/needle_data/3CH-4AA-0005/3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json"
out_params = in_params.replace('.json', '-tuned.json')

mult = np.reshape([13/10, 1], (-1,1))

fbg_needle = FBGNeedle.load_json(in_params)

for k, cal_mat in fbg_needle.cal_matrices.items():
    fbg_needle.cal_matrices[k] = mult * cal_mat
# for

fbg_needle.save_json(out_params)
print(f"Saved updated json w/ scaled calibr. mats by {mult.ravel()} to: \n'{out_params}'")