from flask import Flask
from flask_jsonpify import jsonify
from flask import request, Response
from flask_jsonschema_validator import JSONSchemaValidator
import jsonschema
import json
from machinelearning import runModel

app = Flask(__name__)
JSONSchemaValidator( app = app, root="schemas")

@app.errorhandler( jsonschema.ValidationError )
def onValidationError( e ):
  return Response( "There was a validation error: " + str( e ), 400 )

@app.route("/ai", methods=['POST'])
@app.validate("hepatitis", "model")
def ai():
    obj = json.loads(request.data)
    return jsonify(runModel(obj))
    
app.run(host='0.0.0.0', port='5000')
