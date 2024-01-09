import datetime
import logging
import sys

from dronebuddylib.models.enums import LoggerColors


class Logger:

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)

    def log_error(self, class_name, error_message):
        sys.stdout.write(LoggerColors.RED.value)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys.stdout.write('\n' + current_time + " : ERROR : " + class_name + " : " + error_message + '\n\n')

    def log_info(self, class_name, info_message):
        sys.stdout.write(LoggerColors.YELLOW.value)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys.stdout.write('\n' + current_time + " : INFO :" + class_name + " : " + info_message + '\n\n')

    def log_debug(self, class_name, debug_message):
        sys.stdout.write(LoggerColors.BLUE.value)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys.stdout.write('\n' + current_time + " : DEBUG :" + class_name + " : " + debug_message + '\n\n')

    def log_warning(self, class_name, warning_message):
        sys.stdout.write(LoggerColors.CYAN.value)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys.stdout.write('\n' + current_time + " : WARNING :" + class_name + " : " + warning_message + '\n\n')

    def log_success(self, class_name, success_message):
        sys.stdout.write(LoggerColors.GREEN.value)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys.stdout.write('\n' + current_time + " : SUCCESS :" + class_name + " : " + success_message + '\n\n')
