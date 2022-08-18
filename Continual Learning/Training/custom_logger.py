import sys
import logging

class LoggerWriter(object):
    def __init__(self, writer):
        self._writer = writer
        self._msg = ''

    def write(self, message):
        self._msg = self._msg + message
        while '\n' in self._msg:
            pos = self._msg.find('\n')
            self._writer(self._msg[:pos])
            self._msg = self._msg[pos+1:]

    def flush(self):
        if self._msg != '':
            self._writer(self._msg)
            self._msg = ''
            
def setupLogging(out_path):
    logging.basicConfig(filename=f'{out_path}/log', filemode='w', level=logging.DEBUG, format='%(message)s')
    log = logging.getLogger('logger')
    sys.stdout = LoggerWriter(log.info)