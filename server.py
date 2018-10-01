from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import argparse
import tensorflow as tf #Import tensor flow
import numpy as np
from keras import backend as K
import fingers_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, help='batch size')
args = parser.parse_args()


def load_data(menique,medio,indice,pulgar):

        args = parser.parse_args()
        # Feature columns describe how to use the input.
        (train_x, train_y), (test_x, test_y) = fingers_data.load_data()
        my_feature_columns = []
        for key in train_x.keys():
            print(key)
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

 
 
        # Build 2 hidden layer DNN with 10, 10 units respectively.
        classifier = tf.estimator.DNNClassifier( #Dnn classifier , its backpropagation
            feature_columns=my_feature_columns,
            # 3 hidden layers of 22, 12 and 22 nodes each one.
            hidden_units=[22,12,22],
            # The model must choose between 22 classes, alphabet letters.
            n_classes = 22,
            #el 60 corresponde a una capa oculta con 8 neuronas
            #el 70 corresponde a dos capas ocultas con 22 en la primera y 12 en la segunda
            model_dir='NNFINAL54',
            #Define optimizer
            optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate=0.001,
            l1_regularization_strength=0.001)
        )   


 
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

        dic = {}
        expected = ['a']
        for pred_dict, expec in zip(predictions,expected):
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            dic = {"letter": fingers_data.CLASSES[class_id],"probability":probability*100}
            print(fingers_data.CLASSES[class_id])

        return dic

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
        
        dict = load_data(menique,indice,medio,pulgar)
        letter = dict["letter"]
        probability = dict["probability"]    

        
        message = json.dumps({'result':letter ,"probability":probability}, separators=(',', ':')).encode()
        self.wfile.write(message)




def run():
    server_address = ('', 3000)
    httpd = HTTPServer(server_address, HttpHandler)
    httpd.serve_forever()

run()


