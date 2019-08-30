//**************************************LIBRARIES*******************************
#include <Wire.h> // Include Wire.h - Arduino I2C library
#include <SparkFunMLX90614.h> // Include IR thermometer library

//**************************************GLOBAL VARIABLES*****************************
IRTherm temp; // Create an IRTherm object called temp

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
   temp.begin(); // Initialize I2C library and the MLX90614
   temp.setUnit(TEMP_C); // Set units to Celcius (alternatively TEMP_F or TEMP_K)
   
   Serial.println("CLEARDATA");
   Serial.println("LABEL,TEMPERATURE,MOISTURE");

  // delay(2000);

   }//end of setup

void loop() {

  Serial.print("DATA,TEMPERATURE,MOISTURE,");
  
  //Serial.print("DATA,");
 
  if (temp.read()) // Read from the sensor
  { // If the read is successful:
    //    float ambientT = temp.ambient() // Get updated ambient temperature
    float objectT = temp.object(); // Get updated object temperature
    
    //Serial.println("Ambient: " + String(ambientT));/
    //Serial.println("Object: " + String(objectT));
    Serial.print(String(objectT)+",");
    
//  } //end of if 
  
  int moisture_data=read_moisture();//reads moisture data from moisture sensor
   
  //Serial.println(moisture_data);
  Serial.println(moisture_data);


//   Serial.println("%");

   delay(2000);
   } //end of if 
}//end of loop
