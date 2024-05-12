from flask import Flask, json, g, request, jsonify, json
import vowelizer

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def vowelize():
    json_data = json.loads(request.data)
    vowelizer_instance = vowelizer.Vowelizer()
    response=vowelizer_instance.vowelize(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False,)