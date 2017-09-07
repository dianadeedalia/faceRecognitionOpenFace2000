#necessary importations
from flask import Flask, render_template, redirect, url_for, request
import pickle
#create the app object
app=Flask(__name__)
import face_recognition as fr
#create a route to the home page
@app.route("/", methods=['POST','GET'])
def main():
	if request.method=='GET':
		return render_template("homePage.html")
	else:
		name=request.files["fileUpload"]
		
		return detect_face(name)

def detect_face(name):
	#load the unknown image file
	unknown_img=fr.load_image_file(name)
	unknown_img_enc=fr.face_encodings(unknown_img)
	string=''
	if unknown_img_enc:
		with open("database.pkl","rb") as f:
			(label,enc)=pickle.load(f)
		#time for comparing
		results=fr.compare_faces(enc,unknown_img_enc[0],0.4)
		#print (results)
		#print ("Length of results '{0}'".format(len(results)))
		
		try:
			matching_face_index=results.index(True)
		except ValueError:

			string=str(name) +'not matched!!!'
			message="Try again with a different picture"
		else:
			string=str(name) +'matches'+label[matching_face_index]
			message="Confirmed, Test ended successfully. You may proceed"
	else:
		string="no face detected"
		message="Try again with a different picture"
	return render_template('returnmessage.html',result=message)

@app.route("/tryAgain")
def retry():
	return redirect(url_for('main'))
 

#run the app
if __name__=="__main__":
	app.run(debug=True)

