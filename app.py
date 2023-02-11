from flask import Flask, render_template, request, url_for, make_response
import numpy as np
import pickle


#instance for flask
app = Flask(__name__)

#loading model to web
with open('model_pickle','rb') as f:
	model = pickle.load(f)

#routing to home page
@app.route('/')
def home():
	return render_template('index.html')

#routing to prediction page
@app.route('/predict',methods=['POST', 'GET'])
def predict():
	if request.method == 'POST':
		init_feature = [int(x) for x in request.form.values()]
		final_feature = [np.array(init_feature)]
		prediction = model.predict(final_feature)
		if prediction > 0.5:
			output="yes"
		else:
			output="no"
		if output == "yes":
			return render_template('result.html',prediction_text='{}'.format(output))
		else:
			return render_template('failed.html',prediction_text='{}'.format(output))
				
		"""return render_template('report_sucess.html',prediction_text='{}'.format(output))"""
	if request.method == 'GET':
		return render_template('main1.html')

    
#routing to visualize page
@app.route('/visualize')
def visualize():
	return render_template('viz.html')

#routing to tool page			
@app.route('/tool')
def tool():
	return render_template('tool.html')

#routing to about us page
@app.route('/about')
def about():
	return render_template('about.html')


@app.route('/result',methods=['POST'])
def report():
	if request.method == 'POST':
		name = request.form.get('name')
		return render_template('report_sucess.html', name=name)
	


    
#run and debug the app
if __name__=="__main__":
	app.run(debug=True)



	"""ap_age = request.form["age"]
		ap_income = request.form["income"]
		ap_score = request.form["score"]
		ap_type = request.form["user_type"]
		ap_credit = request.form["credit_his"]
		ap_loan = request.form["loan_amount"]"""