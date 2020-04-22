
import numpy as np
import os
import json
import logging
import pandas as pd

from attribute_feature_explorer import NLPEngine;

from flask import Flask, request, jsonify, url_for
from flask_restplus import Api  #, Resource, fields
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# monkey patch courtesy of
# https://github.com/noirbizarre/flask-restplus/issues/54
# so that /swagger.json is served over https
if os.environ.get('HTTPS'):
    @property
    def specs_url(self):
        """Monkey patch for HTTPS"""
        return url_for(self.endpoint('specs'), _external=True, _scheme='https')

    Api.specs_url = specs_url

api = Api(app, 
        version='2.0', 
        title='Legal Apprentice API',
        description='API for classifing legal sentences',
        )
ns = api.namespace('LegalApprentice', description='ML Model of legal cases')


@ns.route('/Predict/<string:text>')  
class Predict(Resource):
    @ns.doc('use this to predict a sentence classes')
    # @api.expect(prediction_model)
    def get(self,text):
        logging.warning('use this to predict a sentence classes')
        try:


            nlp = NLPEngine();
            nlp.load("version2")

            # data = api.payload
            res = nlp.predict(text,print=False)

            ##result = jsonify(res)
            return res, 200
        except Exception as message:
            res = {
                'hasErrors': True,
                'errorMessage': jsonify(message),
                'payloadCount': 0,
                'payload': []
            }
            ##result = jsonify(res)
            return res, 200

def startup():
    logging.warning('getting started')
    app.run(port=8000, threaded=False, host=('0.0.0.0'))


if __name__ == '__main__':
  startup()
