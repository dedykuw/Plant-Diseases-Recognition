import tensorflow as tf
from keras.models import load_model
# global graph, model, output_list

graph = tf.get_default_graph()
model = load_model('plant_app\AlexNetModel.hdf5')

output_dict = {
               'Jagung - Cercospora Bintik Daun Abu-abu': 7,
               'Jagung - Karat Daun': 8,
               'Jagung - Penyakit hawar daun jagung': 9,
               'Jagung - Healthy': 10,
               }

output_list = list(output_dict.keys())
