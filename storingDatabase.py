import os #for obtaining the images
import face_recognition as fr
import pickle#for storing the lists in a pickle file
path='../Desktop/FRO/faces/db1/known_faces'#file path for the images
imagePaths=os.listdir(path)#obtaining the absolute paths
known_image_encs=[]#for storing the image encodings
known_image_labels=[]#for storing the image labels 
for imagePath in imagePaths:#for each mage

	img=fr.load_image_file(os.path.join(path,imagePath))#for reading the image
	img_enc=fr.face_encodings(img)#for getting their encodings
	if img_enc:#if face was found
		known_image_labels.append(imagePath[:-4])#storing the image labels
		known_image_encs.append(img_enc[0])#storing the image encodings
	else:#if image was not found continue
		continue

with open("database.pkl","wb") as f:#opening the pickle file to write it in byte form
	pickle.dump((known_image_labels,known_image_encs),f)#writing
		