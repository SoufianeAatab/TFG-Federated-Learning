import serial
import struct
import time
import numpy as np

def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except: print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            #port = "COM8";
            return serial.Serial(port, 9600)
        except: print(f"ERROR: Wrong port connection ({port})")
            
def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword: break
        else: print(f'({arduino.port}):',msg, end='')

def getDevices():
    # num_devices = read_number("Number of devices: ")
    print("Hardcoded to 1 device.")
    num_devices = 1

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports: print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]
    return devices

devices = getDevices()

# Send the blank model to all the devices
def receive_model_info(device):
    device.reset_input_buffer()
    [num_layers] = struct.unpack('i', device.read(4))
    print(num_layers)

receive_model_info(devices[0])