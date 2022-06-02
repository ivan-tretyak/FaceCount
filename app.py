from flask import Flask, request, jsonify, abort
from algorithm.main import main, load_model, get_clss


app = Flask(__name__)
print('Model load')
model = load_model()
print('Model loaded')


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal server error"}), 500


@app.route('/', methods=["POST"])
def post():
    print(request.json)
    cls = main(model, request.json)
    data = {'data': {'type': 'people detect', 'class_name':cls[0], 'class_number':cls[1]}}
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
