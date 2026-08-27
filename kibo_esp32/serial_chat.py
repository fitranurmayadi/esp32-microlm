#!/usr/bin/env python3
import serial
import serial.tools.list_ports
import time
import sys
import glob
import threading

def find_esp_port(timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        ports = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
        if ports:
            return ports[0]
        time.sleep(0.2)
    return None

def run_serial_chat(port=None, baudrate=115200):
    if port is None:
        port = find_esp_port()
        
    if port is None:
        print("error: no serial device (/dev/ttyUSB* or /dev/ttyACM*) found", file=sys.stderr)
        return

    print(f"kibo-client: connected to {port} @ {baudrate} baud (Ctrl+C to exit)\n")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1, rtscts=False, dsrdtr=False)
        if "ttyACM" in port:
            ser.dtr = True
            ser.rts = False
        else:
            ser.dtr = False
            ser.rts = False
        time.sleep(0.3)
    except Exception as e:
        print(f"error: failed to open port {port}: {e}", file=sys.stderr)
        return
        
    running = True
    
    def read_from_serial():
        while running:
            try:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
                    sys.stdout.write(data)
                    sys.stdout.flush()
                else:
                    time.sleep(0.01)
            except Exception:
                break
                
    reader_thread = threading.Thread(target=read_from_serial, daemon=True)
    reader_thread.start()
    
    try:
        while running:
            try:
                user_msg = input()
            except EOFError:
                break
                
            if not user_msg.strip():
                continue
                
            msg_to_send = user_msg.strip() + "\n"
            ser.write(msg_to_send.encode('utf-8'))
            ser.flush()
            
    except KeyboardInterrupt:
        print("\nkibo-client: connection closed")
    finally:
        running = False
        time.sleep(0.1)
        try:
            ser.close()
        except:
            pass

if __name__ == "__main__":
    specified_port = sys.argv[1] if len(sys.argv) > 1 else None
    run_serial_chat(specified_port)
