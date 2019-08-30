//**************************************GLOBAL VARIABLES*****************************


int moisture_sensor_pin = A10;//moisture sensor analog pin

int moisture_output_value ;//moisture sensor output data
//*****************************************FUNCTIONS*********************
//This Function is used to read moisture data from the sensor and its return the value 
//in integer form
int read_moisture()
{
   moisture_output_value= analogRead(moisture_sensor_pin);
   moisture_output_value = map(moisture_output_value,550,0,0,100);
   return moisture_output_value;

  
}//end of read_moisture_function 
//***********************************MAIN CODE*********************
void setup() {

   Serial.begin(9600);
   Serial.println("CLEARDATA");
   Serial.println("LABEL,TEMPERATURE,MOISTURE");

  // delay(2000);

   }//end of setup

void loop() {

  Serial.print("DATA,TEMPERATURE,MOISTURE,");
  int moisture_data=read_moisture();//reads moisture data from moisture sensor
   Serial.println(moisture_data);


//   Serial.println("%");

   delay(2000);
}//end of loop
