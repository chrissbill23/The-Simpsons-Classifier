import threading
import time
import sys
def alert(message):
    print("\n")
    for i in range(0, len(message)+4):
        print("#", end="")
    print("\n#", end="")
    for i in range(0, len(message)+2):
        print(" ", end="")
    print("#")
    print("# "+message+" #")
    print("#", end="")
    for i in range(0, len(message)+2):
        print(" ", end="")
    print("#")
    for i in range(0, len(message)+4):
        print("#", end="")
    print("\n")
def stepM(message):
    print("\n")
    print("  \u25bc   "+message+"   \u25bc")
    print("\n")
class ShowTimer:
    def __init__(self):
        self.__stopped = False
    def __show(self):
        while self.__stopped == False:
            sec = str(int(time.clock()) % 60)
            if len(sec) == 1:
                sec = '0'+sec
            min = str((int(time.clock()) // 60) % 60)
            if len(min) == 1:
                min = '0'+min
            h = str((int(time.clock()) // 3600)%24)
            if len(h) == 1:
                h = '0'+h
            sys.stdout.write(" Elapsed time "+h + ":" +min+":"+sec+"\r")
            sys.stdout.flush()
            time.sleep(1)
    def start(self):
        self.__tictac=threading.Thread(target=self.__show, daemon=True)
        self.__tictac.start()
    def stop(self):
        self.__stopped = True