__main-author__ = 'zhengwang'
__co-author__ = 'akshay'

import numpy as np
import threading
import cv2
import pygame
from pygame.locals import *
import socket
import time
import os
import scipy.sparse

pygame.init()
screen=pygame.display.set_mode((70,50))
pygame.mouse.set_visible(0)
temp='[0]'

class ConnectToRPI():
    def __init__(self):
        global temp
        #IP address and Port of RPI
        RPI_TCP_IP = '10.42.0.1'
        RPI_TCP_PORT = 8001
        BUFFER_SIZE = 1024  # Normally 1024, but we want fast response   
        #Make connection to RPI        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((RPI_TCP_IP, RPI_TCP_PORT))
        s.listen(1)
        conn, addr = s.accept()
        print 'Connection address:', addr
        while True:
            conn.send(str(temp))
            time.sleep(0.2)

class CollectData(object):
    
    def __init__(self):
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

        pygame.init()
        self.collect_image()

    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print ('Start collecting images...')
        e1 = cv2.getTickCount()
        #placeholder for images which will be flattened, which is equal to (1,(width x height))
        image_array = np.zeros((1, 38400))
        #image_array = np.zeros((1, 76800))
        label_array = np.zeros((1, 3), 'float')

        # stream video frames one by one
        try:
            global temp
            stream_bytes = ' '
            frame = 1
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
                    
                    # save streamed images  to check whether data collected is valid
                    #cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame),image)
                    
                    #cv2.imshow('roi_image', roi)
                    cv2.imshow('r', roi)
                    cv2.waitKey(1)
                    
                    # reshape the roi image into one row array
                    temp_array = roi.reshape(1, 38400).astype(np.float32)
                    #temp_array = roi.reshape(1, 76800).astype(np.float32)
                    
                    frame += 1
                    total_frame += 1
                    # get input from keypad
                    for event in pygame.event.get():
                        #if keypad is pressed
                        if event.type == KEYDOWN:
                            print("code:"+str(event.key)+"Char:"+chr(event.key))
                            key_input = pygame.key.get_pressed()
                            
                            #if key 'w' is pressed, the robot moves in forward direction.
                            if key_input[pygame.K_w]:
                                print("Forward")
                                saved_frame += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                                #self.ser.write(chr(1))
                                #save images
                                cv2.imwrite('training_images/f/frame{:>05}.jpg'.format(frame),roi)
                                temp='0'
                                
                                
                             #if key 'd' is pressed, the robot moves in right direction.
                            elif key_input[pygame.K_d]:
                                print("Right")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frame += 1
                                #self.ser.write(chr(3))
                                cv2.imwrite('training_images/r/frame{:>05}.jpg'.format(frame),roi)
                                #save images
                                temp='1'
                             #if key 'a' is pressed, the robot moves in left direction.
                            elif key_input[pygame.K_a]:
                                print("Left")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[2]))
                                saved_frame += 1
                                #self.ser.write(chr(4))
                                cv2.imwrite('training_images/l/frame{:>05}.jpg'.format(frame),roi)
                                #save images
                                temp='2'
                            #exit
                            elif key_input[pygame.K_p] or key_input[pygame.K_o]:
                                print ('exit')
                                self.send_inst = False
                                #self.ser.write(chr(0))
                                break
                            else:
                                temp='3'
                            
                                    
                        elif event.type == pygame.KEYUP:
                            temp='3'
                            #self.ser.write(chr(0))

            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]
           
            

            # save training data as a numpy file
            file_name = str(int(time.time()))
            directory = "training_data"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:    
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)

            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print ('Streaming duration:', time0)
            
            
            print(train.shape)
            print(train_labels.shape)
            print ('Total frame:', total_frame)
            print ('Saved frame:', saved_frame)
            print ('Dropped frame', total_frame - saved_frame)

        finally:
            self.connection.close()
            self.server_socket.close()
            cv2.destroyAllWindows()

class RunThread(object):
    def server_thread():
        ConnectToRPI()
    def server_thread2():
        CollectData()
        
    d=threading.Thread(target=server_thread)
    d.start()
    v=threading.Thread(target=server_thread2)
    v.start()
RunThread()    
