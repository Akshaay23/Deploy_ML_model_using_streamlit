import numpy as np 
import pickle


#load the save mode
loaded_model = pickle.load(open("/workspaces/Deploy_ML_model_using_streamlit/trained_model.sav", 'rb'))


input_data=(2,99,60,17,160,36.6,0.453,21)

# chainging input data to numpy_aary
input_data_as_numpy_aaray=np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_aaray.reshape(1,-1)


prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')