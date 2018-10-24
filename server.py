from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import argparse
import tensorflow as tf #Import tensor flow

import fingers_data

def ejecutarServer(menique,medio,indice,pulgar):
  

    (train_x, train_y), (test_x, test_y) = fingers_data.load_data()

    my_feature_columns = []
    for key in train_x.keys():
        print(key)
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier( #Dnn classifier , its backpropagation
        feature_columns=my_feature_columns,
        hidden_units=[22,12,22],
        n_classes = 22,

        model_dir= r'NNFINAL54',
        #Define optimizer
        optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=0.001,
        l1_regularization_strength=0.001)
    )   

    expected = ['a']
    predict_x = {
        'menique': [menique],#Target from thumb
        'medio': [medio],#Target from index finger
        'indice':[indice],#Target from ring finger
        'pulgar': [pulgar],#Target from pinkie

    }
    predictions = classifier.predict(
        input_fn=lambda:fingers_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=100))

    dict = {}

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        dict = {"letter":fingers_data.CLASSES[class_id],"probability":probability}

    print(dict["letter"])
    return dict

class HttpHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/HTML; charset=utf-8')
        self.end_headers()
        
        get_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        menique = get_params["menique"][0]
        medio = get_params["medio"][0]
        indice = get_params["indice"][0]
        pulgar = get_params["pulgar"][0]
        
        menique = float(menique)
        medio= float(medio)
        indice = float(indice)
        pulgar = float(pulgar)
        
        dict = ejecutarServer(menique,indice,medio,pulgar)
        letter = dict["letter"]
        probability = str(dict["probability"])
        print(probability)   

        
        message = json.dumps({'result':letter,'probability':probability}, separators=(',', ':')).encode()
        self.wfile.write(message)




def run():
    server_address = ('', 3000)
    httpd = HTTPServer(server_address, HttpHandler)
    httpd.serve_forever()

run()


