from tflite_runtime.interpreter import Interpreter
from keras.preprocessing.image import img_to_array
import cv2
import copy
import tensorflow as tf

interpreter = Interpreter("tflite_model_64.tflite")

new_model = tf.keras.models.load_model('test_model_7.h5')


def predict(img):
    img = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    img = cv2.resize(img,(width, height))
    img = img_to_array(img)
    img = img / 255
    img = img.reshape(1, width, height, 3)
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()
    flite_results = interpreter.get_tensor(output_details[0]['index'])
    # prediction = np.where(tflite_results < 1, 0, 1)
    out = flite_results[0][0]
    if out < 1:
        out = 0
    else:
        out = 1
    return out


def predict_2(img):
    width, height = 25, 25
    img = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = cv2.resize(img, (width, height))
    img = img_to_array(img)
    img = img / 255
    img = img.reshape(1, width, height, 3)
    return new_model.predict(img)
