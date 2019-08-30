#This code is written by Muntasir FEROZ
#here learned models are loaded to check outputs

import pickle


'''
test=input("DO You Want this?(y/n)")
if(test=='y' or test=='Y'):
    print("HELL YEAH!!!")
    filename=input("Give a file name")
    print(filename+'.sav')
    open('Serialize_Models\ '+filename+'.sav','wb')
else:
    print('BYE BYE MY FRIEND',38,'JOHN')
'''
test_moisture=85
test_temperature = 104.2
test_pulse = 100

print('When Moisture:',test_moisture,'Temperature: ',test_temperature,'PULSE: ',test_pulse)
filename="D:\Myworkplace\Python\health_care\Serialize_Models\ accuracy91_22.sav"
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb')) #model is de-selialized
result = loaded_model.predict([[test_moisture,test_temperature,test_pulse]])
print("The model accuracy91_22.sav says the Patient is ",result)

#----------------------------
filename="D:\Myworkplace\Python\health_care\Serialize_Models\ accuracy88_78.sav"
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb')) #model is de-selialized
result = loaded_model.predict([[test_moisture,test_temperature,test_pulse]])
print("The model accuracy88_78.sav says the Patient is ",result)

#----------------------------
filename="D:\Myworkplace\Python\health_care\Serialize_Models\ accuracy87_32.sav"
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb')) #model is de-selialized
result = loaded_model.predict([[test_moisture,test_temperature,test_pulse]])
print("The model accuracy87_32.sav says the Patient is ",result)