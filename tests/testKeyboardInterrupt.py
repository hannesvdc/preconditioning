import time

try:
    time.sleep(10.0)
except KeyboardInterrupt:  
    print('Exception Caught')
    print('Sleeping Again')
    time.sleep(20.0)
    print('Back Online, Exiting')

