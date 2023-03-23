import serial
import struct
import time
import numpy as np

def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword: break
        else: print(f'({arduino.port}):',msg, end='')