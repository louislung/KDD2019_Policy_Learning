import logging, os

# Create a custom logger

class CustomLogger:
    def __init__(self, name, _log_file='run.log', _stream_level='DEBUG', _file_level='DEBUG'):
        self.logger = logging.getLogger(name)
        self._log_file = _log_file
        self._stream_level = _stream_level.upper()
        self._file_level = _file_level.upper()

        # Check valid input parameter
        # if os.path.exists(self._log_file):
        #     raise Exception('log file {} already exists'.format(self._log_file))
        if self._stream_level not in ['DEBUG','INFO','WARNING','ERROR','CRITICAL']:
            raise Exception('invalid _stream_level '.format(self._stream_level))
        if self._file_level not in ['DEBUG','INFO','WARNING','ERROR','CRITICAL']:
            raise Exception('invalid _file_level '.format(self._file_level))

        # Create handlers
        self.c_handler = logging.StreamHandler()
        self.f_handler = logging.FileHandler(_log_file)
        self.c_handler.setLevel(_stream_level)
        self.f_handler.setLevel(_file_level)

        # Create formatters and add it to handlers
        self.c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        self.f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        self.c_handler.setFormatter(self.c_format)
        self.f_handler.setFormatter(self.f_format)

        # Add handlers to the logger
        self.logger.addHandler(self.c_handler)
        self.logger.addHandler(self.f_handler)
        self.logger.setLevel(logging.DEBUG)

        self.logger.info('Init logger {} to file {}'.format(name, self._log_file))

    def critical(self, message):
        self.logger.critical(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

if __name__ == '__main__':
    log = CustomLogger()
    print(log.logger.handlers)
    log.critical('critical')
    log.error('error')
    log.warning('warning')
    log.info('info')
    log.debug('debug')