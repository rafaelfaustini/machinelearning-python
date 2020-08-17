from flask import Flask
from flask_jsonpify import jsonify
from flask import request, Response
from flask_cors import CORS
from flask_jsonschema_validator import JSONSchemaValidator
import jsonschema
import json
from machinelearning import runModel
from utils import getRandomTest, validateRandomTest

app = Flask(__name__)
CORS(app)
JSONSchemaValidator( app = app, root="schemas")

@app.errorhandler( jsonschema.ValidationError )
def onValidationError( e ):
  return Response( "There was a validation error: " + str( e ), 400 )

@app.route("/ai", methods=['POST'])
@app.validate("image", "model")
def ai():
    obj = json.loads(request.data)
    return jsonify(runModel(obj))

@app.route("/validate", methods=['POST', 'GET'])
def validate():
    if request.method == 'GET':
        return jsonify(getRandomTest())
    elif request.method == 'POST':
        obj = json.loads(request.data)
        print(obj)
        if(obj):
            return jsonify(validateRandomTest(obj))
        else:
            return ''
        
    

    
app.run(host='0.0.0.0', port='5000')
