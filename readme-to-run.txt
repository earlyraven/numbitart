open command line to this folder and run:
gunicorn -w 4 -b 0.0.0.0:$(cat .env | grep FLASK_RUN_PORT | cut -d "=" -f2) app.main:app

Open browser and go to:
http://0.0.0.0:7000/


gunicorn -w 4 -b 0.0.0.0:5000 app.main:app

gunicorn -w 4 -b 0.0.0.0:5000 app.main:app
