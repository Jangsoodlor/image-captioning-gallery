bind = "0.0.0.0:5000"

# Documentations:
# https://gunicorn.org/reference/settings/
# https://flask.palletsprojects.com/en/stable/deploying/gunicorn/

workers = 1
accesslog = "-"
errorlog = "-"

capture_output = False
# change log level
loglevel = "info"
# set to daemon to True for running in background
daemon = False
pidfile = "gunicorn.pid"
