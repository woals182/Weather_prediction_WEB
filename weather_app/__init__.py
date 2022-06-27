from flask import Flask ,render_template, request
import ml

def create_app():

    app = Flask(__name__)

    @app.route('/')
    def index():

        return render_template('index.html'), 200

    @app.route('/weather',methods=["POST"])
    def prediction():
        if request.method == "POST":
            month = request.form['month']
            day = request.form['day']
            hours = request.form['hour']

            M = int(month)
            D = int(day)
            hour = int(hours)

            def doy(M,D):
                K = 2
                N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
                return N

            N=doy(M,D)
            y_pred = ml.clf.predict([[N,2022,hour]])
    
            pred = f"""2022년 {M}월{D}일의 {hour}시의 날씨
            기온 :  {round(float(y_pred[0][0]),2)}(°C),
            강수량 :  {round(float(y_pred[0][1]),2)}(mm),
            풍속 :  {round(float(y_pred[0][2]),2)}(m/s),
            풍향 :  {round(float(y_pred[0][3]),2)}(16방위),
            습도 :  {round(float(y_pred[0][4]),2)}(%),
            현지기압 :  {round(float(y_pred[0][5]),2)}(hPa)"""

            return render_template('index.html', predict=pred) , 200

    return app
