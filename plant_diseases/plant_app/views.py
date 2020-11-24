from django.shortcuts import render
from keras.preprocessing import image
from django.http import JsonResponse
import numpy as np
from .deeplearning import graph, model, output_list
import base64

def index_exists(ls, i):
    return (0 <= i < len(ls)) or (-len(ls) <= i < 0)
def index(request):
    if request.method == 'POST' and request.FILES.get('myfile'):
        myfile = request.FILES['myfile']
        b64_img = base64.b64encode(myfile.file.read()).decode('ascii')
        img = image.load_img(myfile, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255

        with graph.as_default():
            prediction = model.predict(img)

        prediction_flatten = prediction.flatten()
        max_val_index = np.argmax(prediction_flatten)
        print(max_val_index)
        if max_val_index > 6 and max_val_index < 11 :
            result = output_list[max_val_index]
        else:
            result = "undefined"
        return JsonResponse({
            'result': result})

    return render(request, "plant_app/index.html")
