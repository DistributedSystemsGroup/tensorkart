#!/usr/bin/env python

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from utils import take_screenshot, prepare_image
from utils import XboxController
import tensorflow as tf
import model
import threading
from termcolor import cprint
import json
import random
PORT_NUMBER = 8082

# Start session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

gas = 0
# Load Model
saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")

# Init contoller for manual override
real_controller = XboxController()

class ControllerState(object):
    def __init__(self):
        self.START_BUTTON = 0
        self.Z_TRIG = 0
        self.B_BUTTON = 0
        self.A_BUTTON = 0
        self.R_TRIG = 0
        self.L_TRIG = 0
        self.X_AXIS = 0
        self.Y_AXIS = 0

def amp(steer):
    threshold = 20
    gain = 1.8
    # threshold = 25
    # gain = 1.9
    if steer > threshold:
        steer = max(gain*steer, 80)
    elif steer < -threshold:
        steer = max(gain*steer, -80)
    # else:
    #     steer *= 0.5
    return int(steer)

# Play
class myHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global gas
        ### determine manual override
        manual_override = real_controller.manual_override()

        if (not manual_override):

            ## Look
            image = take_screenshot()
            vec = prepare_image(image)


            ## Think
            joystick = model.y.eval(session=sess, feed_dict={model.x: [vec], model.keep_prob: 1.0})[0]
            gas = 1 - gas
            joystick[2] =  gas # slow down a bit

        else:
            joystick = real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)


        ## Act

        steer = int(joystick[0] * 80)
        ####### amplification
        steer = amp(steer)

        output = [
            steer, # LEFT RIGHT ANALOG
            int(joystick[1] * 80), #  UP DOWN ANALOG
            int(round(joystick[2])), # A
            int(round(joystick[3])), # X
            int(round(joystick[4])), # RB
        ]

        ### print to console
        if (manual_override):
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')
        # TODO: include other buttons as in controller.c (input-bot)
        json_output = ControllerState()
        json_output.X_AXIS = steer
        json_output.Y_AXIS = int(joystick[1] * 80)
        json_output.A_BUTTON = int(round(joystick[2]))

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(json.dumps(json_output.__dict__))
        return


if __name__ == '__main__':
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
    thread = threading.Thread(target=server.serve_forever, args=())
    thread.daemon = True
    thread.start()
    raw_input('Serving now... press <Enter> to shut down.')
    print 'Shutting down...'
