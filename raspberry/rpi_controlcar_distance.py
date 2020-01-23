
__author__ = 'akshay'

import socket
import time
import RPi.GPIO as GPIO

GPIO.setwarnings(False)

# create a socket and bind socket to the host
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.42.0.1', 8001))
buffe=1024

def measure():
    """
    measure distance
    """
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    start = time.time()

    while GPIO.input(GPIO_ECHO)==0:
        start = time.time()

    while GPIO.input(GPIO_ECHO)==1:
        stop = time.time()

    elapsed = stop-start
    distance = (elapsed * 34300)/2

    return distance




# referring to the pins by GPIO numbers
GPIO.setmode(GPIO.BCM)

# define pi GPIO
GPIO_TRIGGER = 23
GPIO_ECHO    = 24
l1=21
l2=20
r1=16
r2=12

# output pin: Trigger
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
# input pin: Echo
GPIO.setup(GPIO_ECHO,GPIO.IN)
GPIO.setup(l1,GPIO.OUT)
GPIO.setup(l2,GPIO.OUT)
GPIO.setup(r1,GPIO.OUT)
GPIO.setup(r2,GPIO.OUT)
# initialize trigger pin to low
GPIO.output(GPIO_TRIGGER, False)
GPIO.output(l1, False)
GPIO.output(l2, False)
GPIO.output(r1, False)
GPIO.output(r2, False)

try:
    while True:
        distance = measure()
        #print "Distance : %.1f cm" % distance
        # send data to the host every 0.5 sec
        #client_socket.send(str(distance))
        time.sleep(0.01)
        data=client_socket.recv(buffe)
        print 'prediction:',data[0],'distance:',distance
        if distance >10:
            if data[0]=='0':
                GPIO.output(l1, False)    #forward
                GPIO.output(l2, True)
                GPIO.output(r1, False)
                GPIO.output(r2, True)
            elif data[0]=='1':          #right
                GPIO.output(l1, True)
                GPIO.output(l2, False)
                GPIO.output(r1, False)
                GPIO.output(r2, True)
            elif data[0]=='2':
                GPIO.output(l1, False)   #left
                GPIO.output(l2, True)
                GPIO.output(r1, True)
                GPIO.output(r2, False)
            else:
                GPIO.output(l1, False)
                GPIO.output(l2, False)
                GPIO.output(r1, False)
                GPIO.output(r2, False)
        else:
            GPIO.output(l1, False)
            GPIO.output(l2, False)
            GPIO.output(r1, False)
            GPIO.output(r2, False)
finally:
    client_socket.close()
    GPIO.cleanup()
    client_socket.close()
