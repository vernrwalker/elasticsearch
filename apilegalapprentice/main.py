from flask import Flask
from flask_restplus import Resource, Api

# http://michal.karzynski.pl/blog/2016/06/19/building-beautiful-restful-apis-using-flask-swagger-ui-flask-restplus/

app = Flask(__name__)                  #  Create a Flask WSGI application
api = Api(app)                         #  Create a Flask-RESTPlus API

@api.route('/hello')                   #  Create a URL route to this resource
class HelloWorld(Resource):            #  Create a RESTful resource
    def get(self):                     #  Create GET endpoint
        return {'hello': 'world'}

if __name__ == '__main__':
    app.run(debug=True)  