import serial
import struct
import time
import numpy as np
from serial.tools.list_ports import comports

def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except: print("ERROR: Not a number")

def read_port(msg, available_ports):
    while True:
        try:
            port = input(msg)
            #index = input(msg)
            #port = "COM8";
            return serial.Serial(port, 9600)
        except: print(f"ERROR: Wrong port connection ({available_ports[index-1]})")
            
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
    for i,available_port in enumerate(available_ports): print(f"{available_port}")

    devices = [read_port(f"Port device_{i+1}: ", available_ports) for i in range(num_devices)]
    return devices

# Send the blank model to all the devices
def receive_model_info(device):
    device.reset_input_buffer()
    device.write(b's') # Python --> ACK --> Arduino
    print_until_keyword('start', device) # CLEAN SERIAL
    
    bytesToRead = device.read(1).decode()
    time.sleep(1)
    if bytesToRead == 'i':
        [num_layers] = struct.unpack('i', device.read(4))
        dimms = []
        for i in range(num_layers):
            [rows, cols] = struct.unpack('ii', device.read(8))
            dimms.append((1,cols)) # bias
            dimms.append((rows,cols)) # matrix weigths
    return num_layers, dimms

def initialize_device_weights(device, layer, bias_dimm, w_dimm):
    bias = np.zeros(bias_dimm)
    weights = np.random.randn(w_dimm[0], w_dimm[1]) * np.sqrt(6.0 / (w_dimm[0] + w_dimm[1]))
    print(f"Sending weights for LAYER {layer}")
    for b in bias.reshape(-1):
        data = device.read()
        device.write(struct.pack('f', b))

    for w in weights.reshape(-1):
        data = device.read()
        device.write(struct.pack('f', w))
    
devices = getDevices()
num_layers, dimms = receive_model_info(devices[0])
    
for device in devices:
    for i in range(0,len(dimms),2):
        initialize_device_weights(device,i//2,dimms[i], dimms[i+1])

from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

X,y = load_digits(return_X_y=True)
onehot_encoder = OneHotEncoder(sparse=False)
y_train_one_hot = onehot_encoder.fit_transform(y.reshape(-1,1))

def sent_sample(device, X, y):
    for s in X.reshape(-1):
        data = device.read()
        device.write(struct.pack('f', s))

    for t in y.reshape(-1):
        data = device.read()
        device.write(struct.pack('f', t))

def get_tick():
    return round(time.time() * 1000)

def main(device):
    losses = []
    epochs = 100
    for epoch in range(epochs):
        error = 0.0
        dts = 0
        start = get_tick()
        for i in range(1000):
            device.write(b"t")
            sent_sample(device, X[i], y_train_one_hot[i])
            dt = device.read(4)
            [dt] = struct.unpack('i', dt)
            n_error = device.read(4)
            [loss] = struct.unpack('f', n_error)
            error += loss
            dts += dt
        end = get_tick()
        print(f"{epoch}/{epochs} => {error/1000.0}, {end-start}ms")
        losses.append(error / 1000.0)

main(devices[0])

print("Model sent")
