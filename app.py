from flask import Flask, request, render_template
import model

app = Flask(__name__)

@app.route('/classify', method=['GET'])
def classify_type():
    try:
        distance     = request.args.get('distance')
        haversine    = request.args.get('haversine')
        pickup_hour  = request.args.get('phour')
        pickup_min   = request.args.get('pmin')
        dropoff_hour = request.args.get('dhour')
        dropoff_min  = request.args.get('dmin')
        temp         = request.args.get('temp')
        humid        = request.args.get('humid')
        solar        = request.args.get('solar')
        dust         = request.args.get('dust')

        attribute = np.array([distance,haversine,pickup_hour,pickup_min,dropoff_hour,dropoff_min,temp,humid,solar,dust]).reshape(1,-1)

        if attribute.shape == (1,10):
            print("attributes valid")
            #Get the output from the classification model
            Duration = model.predict_duration(attributes=attribute)

        # Render the otuput in new HTML Page

        return render_template('output.html',Duration=Duration)
    except:
        return 'Error'

if __name__ == '__main__':
    app.run(debug=True)