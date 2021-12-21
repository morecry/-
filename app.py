from flask import Flask
import json

from inference import API

app = Flask(__name__)
api = API()

@app.route("/")
def hello_world():
    return app.send_static_file('index.html')

@app.route('/query_api/<text>')
def query_api(text):
    print(text)
    results, values = api.query(text)
    ret = []
    for r, v in zip(results, values):
        ret.append({
            'text': r,
            'score': v
        })
    return json.dumps(ret)