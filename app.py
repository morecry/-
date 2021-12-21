from flask import Flask
import json

from inference import API

app = Flask(__name__)
api = API()

@app.route("/")
def hello_world():
    return app.send_static_file('index.html')

@app.route('/query_api/<text>/<query_type>')
def query_api(text, query_type):
    print(text)
    results, values = api.query(text, type=query_type)
    ret = []
    for r, v in zip(results, values):
        ret.append({
            'text': r,
            'score': v
        })
    return json.dumps(ret)