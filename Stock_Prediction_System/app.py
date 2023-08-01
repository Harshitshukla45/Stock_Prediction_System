import matplotlib
matplotlib.use('Agg')
from flask import Flask , render_template , current_app, url_for
from stock_class import STOCK
app = Flask(__name__)


def get_object(stock_symbol):
    if stock_symbol == 'AAPL':
        if not hasattr(current_app, 'APPL'):
            current_app.APPL = STOCK('AAPL')
        return current_app.APPL
    elif stock_symbol == 'GOOGL':
        if not hasattr(current_app, 'GOOGL'):
            current_app.GOOGL = STOCK('GOOGL')
        return current_app.GOOGL
    elif stock_symbol == 'MSFT':
        if not hasattr(current_app, 'MSFT'):
            current_app.MSFT = STOCK('MSFT')
        return current_app.MSFT


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/apple')
def apple():
    APPL = get_object('AAPL')
    FORECASTED_DATA = url_for('static', filename='AAPL_FORECASTED_DATA.png')
    PLOTTED_DATA = url_for('static', filename='AAPL_PLOTTED_DATA.png')
    PREDICTED_DATA = url_for('static', filename='AAPL_PREDICTED_DATA.png')
    return render_template('index2.html',title='APPLE',FORECASTED_DATA=FORECASTED_DATA,PLOTTED_DATA=PLOTTED_DATA,PREDICTED_DATA=PREDICTED_DATA)

@app.route('/google')
def google():
    GOOGL = get_object('GOOGL')
    FORECASTED_DATA = url_for('static', filename='GOOGL_FORECASTED_DATA.png')
    PLOTTED_DATA = url_for('static', filename='GOOGL_PLOTTED_DATA.png')
    PREDICTED_DATA = url_for('static', filename='GOOGL_PREDICTED_DATA.png')
    return render_template('index2.html',title='GOOGLE',FORECASTED_DATA=FORECASTED_DATA,PLOTTED_DATA=PLOTTED_DATA,PREDICTED_DATA=PREDICTED_DATA)


@app.route('/microsoft')
def microsoft():
    MSFT = get_object('MSFT')
    FORECASTED_DATA = url_for('static', filename='MSFT_FORECASTED_DATA.png')
    PLOTTED_DATA = url_for('static', filename='MSFT_PLOTTED_DATA.png')
    PREDICTED_DATA = url_for('static', filename='MSFT_PREDICTED_DATA.png')
    return render_template('index2.html',title='MICROSOFT',FORECASTED_DATA=FORECASTED_DATA,PLOTTED_DATA=PLOTTED_DATA,PREDICTED_DATA=PREDICTED_DATA)

if __name__ == "__main__":
    app.run(threaded=True)