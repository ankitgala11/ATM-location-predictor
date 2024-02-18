from flask import Flask, render_template, request, redirect, url_for
from model import model
import json

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/location/", methods=['GET', 'POST'])
@app.route("/location", methods=['GET', 'POST'])
def location():    
    if request.method == 'POST':
        form = request.form
        print(form)
        location = form['location']
        result = []
        
        result = model(location, False)

        result = result.to_html(classes='table table-bordered border-5 table-hover align-middle text-center m-0', index=False)
        return render_template('location.html', location=location, result=result)
    return redirect(url_for('index'))

@app.route("/location/<location>", methods=['GET'])
def location2(location):
    if(location):
        return render_template('location.html', location=location)
    return redirect(url_for('index'))



@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/poimap')
def poimap():
    return render_template('poimap.html')

@app.route('/footmap')
def footmap():
    return render_template('footmap.html')

@app.route('/refresh')
def refresh():
    with open('result_json.json', 'w') as outfile:
        json.dump({"location":"", "data":[]}, outfile)
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(debug=True)