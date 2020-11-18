import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# global graph, model, output_list

graph = tf.get_default_graph()
folder = Path("plant_app/")
filetoOpen = folder / "AlexNetModel.hdf5"
model = load_model(str(filetoOpen))

output_dict = {
               'Jagung - Cercospora Bintik Daun Abu-abu': 7,
               'Jagung - Karat Daun': 8,
               'Jagung - Penyakit hawar daun jagung': 9,
               'Jagung - Healthy': 10,
               }

output_list = list(output_dict.keys())
