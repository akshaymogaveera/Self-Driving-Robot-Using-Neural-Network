__author__ = 'akshay'
__co-author__ = 'zhengwang'

import glob
import numpy as np
import cv2
from pygame.locals import *
import socket
import time
import threading
import tensorflow as tf
n_nodes_hl1 = 32
n_classes = 3
temp_prediction='0'

class SendToRPI(object):
    def __init__(self):
        global temp_prediction
        TCP_IP = '10.42.0.1'
        TCP_PORT = 8001
        BUFFER_SIZE = 1024  # Normally 1024, but we want fast response
                 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        conn, addr = s.accept()
        print 'Connection address:', addr
        while True:
            #Send prediction made by Neural Network to RPI
            conn.send(str(temp_prediction.strip('[]')))# echo
            time.sleep(0.2)

        conn.close()


        
class MakePrediction(object):

    def __init__(self):

        self.x = tf.placeholder('float', [1,38400])
        self.y = tf.placeholder('float',[1,3])

        self.server_socket = socket.socket()
        self.server_socket.bind(('10.42.0.1', 8000))
        self.server_socket.listen(0)

        # accept a single connection
        self.connection = self.server_socket.accept()[0].makefile('rb')
        

        self.send_inst = True
        
        # create labels
        self.k = np.zeros((3, 3), 'float')
        for i in range(3):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 3), 'float')
        self.collect_image()
        

        
    
    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print ('Start collecting images...')
        e1 = cv2.getTickCount()
        #image_array = np.zeros((1, 76800))
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 3), 'float')
        frame=1
        # stream video frames one by one
        try:
            global temp_prediction
            prediction = self.neural_network_model(self.x)
            sess=tf.Session()    
            # OLD:
            #sess.run(tf.initialize_all_variables())
            # NEW:
            sess.run(tf.global_variables_initializer())
            saver=tf.train.Saver()
            #saver=tf.train.import_meta_graph('/home/akshay/Downloads/savedata/project.meta')
            saver.restore(sess,'/home/akshay/Downloads/savedata/project')  
                
            stream_bytes = ' '
            while self.send_inst:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    
                    # select lower half of the image
                    roi = image[0:120, :]
                    #roi=image
                    # save streamed images 
                    #cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame),image)
                    
                    #cv2.imshow('roi_image', roi)
                    cv2.imshow('image',roi)
                    cv2.waitKey(1)
                    
                    # reshape the roi image into one row array
                    #temp_array = roi.reshape(1,76800).astype(np.float32)
                    temp_array = roi.reshape(1,38400).astype(np.float32)
                    if frame%3 is 0 :
                        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                        p1=tf.argmax(prediction,1)
                        #save prediction to global variable
                        temp_prediction=p1.eval({self.x:temp_array},session=sess)
                        temp_prediction=str(temp_prediction)
                        #print prediction.eval({x:tempimage})

                    frame += 1
        finally:
            self.connection.close()
            self.server_socket.close()
            #cv2.closeAllWindows()
            #sess.close()
            #ss.brea()


    def neural_network_model(self,data):
        
        hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([38400, n_nodes_hl1],stddev=1.0)),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
        output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_classes],stddev=1.0)),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}


        l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
        output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']

        return output


        
class ThreadServer(object):
    def server_thread():
        SendToRPI()
    def server_thread2():
        MakePrediction()
        
    process1=threading.Thread(target=server_thread)
    process1.start()
    process2=threading.Thread(target=server_thread2)
    process2.start()
ThreadServer()    
