Open command line to this folder and run:
bash runme.sh
(this will run the following:  gunicorn -w 4 -b 0.0.0.0:5000 app.main:app)

Open browser and go to:
http://0.0.0.0:5000/
or
http://localhost:5000/
