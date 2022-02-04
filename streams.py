from constants import CKeys, CColors
import zmq
import numpy as np
import datetime
import sys
from pdb import set_trace
import os

class DataFrame():
    def __init__(self, tvec = np.array([0,0,0]), rvec = np.array([0,0,0]), kfErrorCov = None, timestamp = None, framenumber = 0):
        
        # Make them arrays if they're lists
        if isinstance(tvec, list):
            tvec = np.array(tvec)
        if isinstance(rvec, list):
            tvec = np.array(rvec)
        
        self.tvec = tvec
        self.rvec = rvec
        self.kfErrorCov = kfErrorCov
        self.timestamp = timestamp
        self.framenumber = framenumber

    def dict(self):
        data_dict = {
            CKeys.TVEC : self.tvec,
            CKeys.RVEC : self.rvec,
            CKeys.KF_ERROR_COV : self.kfErrorCov,
            CKeys.TIME_STAMP : self.timestamp,
            CKeys.FRAME_NUMBER : self.framenumber
        }
        return data_dict

class Publisher():
    def __init__(self, socket=None, context=None):
          
        if socket is not None:
            self.connect(socket, context=context)

    def connect(self, socket="tcp://localhost:5555", context=None):
        # If a number was input for socket, assume
        # it specifies a port number on localhost    
        if type(socket) == (int or float):
            socket = "tcp://localhost:%d" % socket
        elif socket.isnumeric():
            socket = "tcp://localhost:%d" % int(socket)

        # ZMQ context (only need one per program)
        if context is None:
            self.context = zmq.Context()
        else:
            self.context = context
        
        # Set up ZMQ connection
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.connect(socket)
        self.socket_str = socket
    
    def send_frame(self, data = None):
        if data is not None:
            if not isinstance(data, dict):
                msg = data.dict()
            else:
                msg = data
            self.socket.send_pyobj(msg)
        else:
            program_name = os.path.split(sys.argv[0])[-1]
            print(CColors.WARNING + "<%s> Warning: send_frame called without any data. Nothing to do." % program_name + CColors.ENDC)


class Subscriber():
    def __init__(self, socket=None, context=None):

        if socket is not None:
            self.connect(socket=socket, context=context)
    
    def connect(self, socket="tcp://*:5555", context=None):
        # If a number was input for socket, assume
        # it specifies a port number on localhost    
        if type(socket) == (int or float):
            socket = "tcp://*:%d" % socket
        elif socket.isnumeric():
            socket = "tcp://*:%d" % int(socket)

        # ZMQ context (only need one per program)
        if context is None:
            self.context = zmq.Context()
        else:
            self.context = context

        # Set up socket for receiving images
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(socket)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'') # This sets filter to "accept everything
        self.socket.setsockopt(zmq.CONFLATE, True)
    
    def receive_frame(self, blocking = True):
        if blocking is True:
            zmqFlags = 0
        else:
            zmqFlags = zmq.NOBLOCK
        try:
            msg = self.socket.recv_pyobj(flags=zmqFlags)
            data = DataFrame()
            data.tvec = msg[CKeys.TVEC]
            data.rvec = msg[CKeys.RVEC]
            data.kfErrorCov = msg[CKeys.KF_ERROR_COV]
            data.timestamp = msg[CKeys.TIME_STAMP]
            data.framenumber = msg[CKeys.FRAME_NUMBER]

        except zmq.error.ZMQError as err:
            data = None
        except Exception as err:
            program_name = os.path.split(sys.argv[0])[-1]
            print(CColors.FAIL + "<%s> ERROR: Deserialization error" % program_name)
            print(CColors.FAIL + str(err) + CColors.ENDC)
            sys.exit()
        
        return data
        
