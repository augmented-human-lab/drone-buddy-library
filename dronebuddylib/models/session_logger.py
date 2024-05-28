from datetime import datetime


class SessionLogger:

    def __init__(self, logger_file_location: str):
        self.file_location = logger_file_location + str(int(datetime.now().timestamp())) + ".txt"
        self.file = open(self.file_location, "w")
        self.file.write("Session started at: " + str(datetime.now()) + "\n")

    def log_chat(self, role, token_count, message):
        self.file.write(
            str(datetime.now()) + ": token count-"
            + str(token_count) + " : role - " + role + " : message - " + message + "\n")

    def close_file(self):
        self.file.write("Session ended at: " + str(datetime.now()) + "\n")
        self.file.close()
