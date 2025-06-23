'''
Livestream Receiver Class Definition

Description: Receiver class for live BCI data streaming.

Joseph Hong

'''
# =============================================================
# =============================================================
# INCLUDES

from socket import *
from struct import *
import numpy as np
import mne
from datetime import datetime
from broadcasting import TCP_Server
import threading


#==========================================================================================
#==========================================================================================
# CLASS DEFINITIONS

# Event marker
class Marker:
    def __init__(self):
        self.position = 0 
        self.points = 0
        self.channel = -1
        self.type = ""
        self.description = ""

# LivestreamReceiver
class LivestreamReceiver:
    # Initialization, takes connection IP and port number as arguments
    def __init__(self, address="169.254.1.147", port=51244, broadcast = False,):
        self.address = address
        self.port = port
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.channel_count = 60
        self.sampling_interval_us = 1000000
        self.sampling_frequency = 1000
        self.resolutions = []
        self.channel_names = []
        self.info = None
        self.last_block = -1
        self.first_packet = True
        self.data = np.zeros((self.channel_count,0))
        
        # For sending classification results to different application via network interface
        self.broadcasting = broadcast
        self.server = None

        # If broadcasting classification (for further applications), set up UDP server
        if broadcast:
            self.server = TCP_Server.TCPServer()
            self.server.initialize_connection()
            print("Server Initialized")
            # Start server in a background thread
            # threading.Thread(target=self.server.initialize_connection, daemon=True).start()

            # Run GUI in main thread
            # gui = TCP_Server.ServerGUI(self.server)
            # gui.root.mainloop()

    # Read Data from Connection
    def recv_data(self, requested_size):
        return_stream = b''     # Byte string for Python 3
        while len(return_stream) < requested_size:
            data_bytes = self.socket.recv(requested_size - len(return_stream))
            if not data_bytes:      # Byte string comparison
                raise RuntimeError("Connection broken")     # Updated exception syntax
            return_stream += data_bytes
        return return_stream
    
    # Decode Data to String
    def split_string(self, raw):
        stringlist = []
        s = ""
        for i in range(len(raw)):
            if raw[i:i+1] != b'\x00':  # Byte string comparison
                s = s + raw[i:i+1].decode('utf-8')  # Decoding byte to string
            else:
                stringlist.append(s)
                s = ""
        return stringlist
    
    # Initialize variables
    def get_properties(self,rawdata):
        self.channel_count, self.sampling_interval_us = unpack('<Ld', rawdata[:12])
        self.resolutions = [unpack('<d', rawdata[12 + i * 8:20 + i * 8])[0] for i in range(self.channel_count)]
        self.channel_names = self.split_string(rawdata[12 + 8 * self.channel_count:])
        self.sampling_frequency = (1 / self.sampling_interval_us) * 1000000
        ch_types = ["eeg"] * self.channel_count
        self.info = mne.create_info(self.channel_names, ch_types=ch_types, sfreq=self.sampling_frequency)
        self.info.set_montage("standard_1020")
        print(f"Channels: {self.channel_count}, Sampling interval: {self.sampling_interval_us} us")
        print(f"Sampling frequency: {self.sampling_frequency} Hz")
        print(f"Channel names: {self.channel_names}")

    # Unpack data packets
    def unpack_data(self, rawdata):
        (block, points, marker_count) = unpack('<LLL', rawdata[:12])

        data = []
        for i in range(points * self.channel_count):
            index = 12 + 4 * i
            value = unpack('<f', rawdata[index:index+4])
            data.append(value[0])

        markers = []
        index = 12 + 4 * points * self.channel_count
        for m in range(marker_count):
            markersize = unpack('<L', rawdata[index:index+4])

            ma = Marker()
            (ma.position, ma.points, ma.channel) = unpack('<LLl', rawdata[index+4:index+16])
            typedesc = self.split_string(rawdata[index+16:index+markersize[0]])
            ma.type = typedesc[0]
            ma.description = typedesc[1]

            markers.append(ma)
            index += markersize[0]

        return (block, points, marker_count, data, markers)

    # AFTER INITIALIZATION: Get Raw EEG Data
    def get_data(self):
        # Read GU.ID and the message type
        rawhdr = self.recv_data(24)
        (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)
        rawdata = self.recv_data(msgsize - 24)

        # If EEG data block, unpack and return data
        if msgtype == 4:
            (block, points, markerCount, data, markers) = self.unpack_data(rawdata)

            # Check for overflow, where a few blocks of data were skipped
            if self.last_block != -1 and block > self.last_block + 1:
                print("*** Overflow with " + str(block - self.last_block) + " datablocks ***")
            # Update last block
            self.last_block = block

            # If there are event markers, display them
            if markerCount > 0:
                for m in range(markerCount):
                    print("===================================")
                    print("Marker " + markers[m].description + " of type " + markers[m].type)
                    print("===================================")
                    print("")
                    # self.server.send_message_tcp("MARKER")

            # Get voltage unit 
            data = np.array(data)
            data = data * self.resolutions[0]

            # Reshape the array to convert it to a two-dimensional array of channels x timepoints
            data = data.reshape((self.channel_count, points), order='F')  # 'F' for column-major order
            return data

        # If ending message, stop the connection
        elif msgtype == 3:
            print("Stopped.")
  
    # Initialize Connection. sfreq, ch_names, numChannels, dataBuffer = bci.InitializeConnection()
    def initialize_connection(self):
        self.socket.connect((self.address, self.port))
        print(f"Connected to {self.address}:{self.port}", datetime.now())

        rawhdr = self.recv_data(24)
        id1, id2, id3, id4, msgsize, msgtype = unpack('<llllLL', rawhdr)
        rawdata = self.recv_data(msgsize - 24)

        # If handshake packet, initialize the variables of the object
        if msgtype == 1:
            self.get_properties(rawdata)
            self.first_packet = False
        else:
            raise RuntimeError("Initialization message not received")
        
        self.data = np.ones((self.channel_count,0))

        print("===================================")
        print("Initial Data Shape:", np.shape(self.data))
        
        return self.sampling_frequency, self.channel_names, self.channel_count, self.data

    # Use Classification
    def use_classification(self, prediction):
        if prediction == 0:
            print("Rest")
        elif prediction == 1:
            if self.broadcasting:
                self.server.send_message_tcp("TAP")
        else:
            print("??")

    # Disconnect
    def disconnect(self):
        self.socket.close()
        print("Connection closed.")