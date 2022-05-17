# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request, render_template, jsonify
import os

file_path = os.getcwd()
app = Flask('__name__', template_folder=file_path + './templates')

# @app.route('/<name>')
# def index(name):
#     return 'Hi {}'.format(name)

@app.route('/', methods=['GET', 'POST'])
def squared():
    if request.method == 'POST':
        input_value = request.form.get('user_input',False)
        _output = float(input_value)*float(input_value)
        
    return render_template('index.html', output=_output)

if __name__ == '__main__':
    app.run(debug=True)
    