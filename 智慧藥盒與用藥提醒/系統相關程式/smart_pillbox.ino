
#include <AccelStepper.h>
#include <Bounce2.h>
#include <Wire.h>    // for I2C with RTC module
#include "RTClib.h"  //to show time
#include <EasyBuzzer.h>
#include <AsyncTimer.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "EEPROM.h"
#include <Preferences.h>

#define EEPROM_SIZE 64
#define IN1 19
#define IN2 18
#define IN3 5
#define IN4 17
#define IN_CABINET_SWITCH 4
#define BEEP_ITERVAL 1000                //蜂鳴器多久響一次
#define BEEP_DURATION 500                //蜂鳴器一次響多久
#define PICKER_PILL_FAIL_DURATION 15000  //多久後確定真的未取藥
#define WIF_SSID "AdamGalaxyA71"
#define WIFI_PASS "aqz02040204"

char ssid[] = WIF_SSID;   // your SSID
char pass[] = WIFI_PASS;  // your SSID Password

unsigned short intervalCheckRunId = 0;
Preferences preferences;
int motorPosition = 0;
int cabinetStatus;
int isWaitTakePill = 0;
uint32_t eeMorning = 0, eeNoon = 0, eeNight = 0, eeTime = 0;
AccelStepper stepper(4, IN1, IN3, IN2, IN4);
Bounce2::Button buttonPauseEx = Bounce2::Button();
RTC_DS3231 rtc;
AsyncTimer timerForBeep;
WiFiClient espClient;
PubSubClient mqtt_client(espClient);
char daysOfTheWeek[7][12] = { "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
String tempHandlerTime = "";

void readConfig() {
  eeMorning = preferences.getUInt("morning", 0);
  eeNoon = preferences.getUInt("noon", 0);
  eeNight = preferences.getUInt("night", 0);
  eeTime = preferences.getUInt("time", 0);
}

String getValue(String data, char separator, int index) {
  int found = 0;
  int strIndex[] = { 0, -1 };
  int maxIndex = data.length() - 1;

  for (int i = 0; i <= maxIndex && found <= index; i++) {
    if (data.charAt(i) == separator || i == maxIndex) {
      found++;
      strIndex[0] = strIndex[1] + 1;
      strIndex[1] = (i == maxIndex) ? i + 1 : i;
    }
  }

  return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived topic = ");
  Serial.print(topic);
  Serial.println();

  Serial.print("payload = ");
  char data[50];
  memcpy(data, payload, length);
  data[length] = 0;
  Serial.println(data);

  int indicator = memcmp((const char*)topic, "/pillbox/RTC", strlen("/pillbox/RTC"));
  if (indicator == 0) {
    Serial.print("已校正時間為:");


    //rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));
    String year = getValue(data, ' ', 0);
    String month = getValue(data, ' ', 1);
    String day = getValue(data, ' ', 2);
    String hour = getValue(data, ' ', 3);
    String minute = getValue(data, ' ', 4);
    String second = getValue(data, ' ', 5);
    rtc.adjust(DateTime(atoi(year.c_str()), atoi(month.c_str()), atoi(day.c_str()), atoi(hour.c_str()), atoi(minute.c_str()), atoi(second.c_str())));

    DateTime now = rtc.now();
    uint32_t timestamp = now.unixtime();
    preferences.putUInt("time", timestamp);
    readConfig();
    mqtt_client.publish("/pillbox", "ok");
  }

  indicator = memcmp((const char*)topic, "/pillbox/morning", strlen("/pillbox/morning"));
  if (indicator == 0) {
    preferences.putUInt("morning", atol(data));
    readConfig();
    mqtt_client.publish("/pillbox", "ok");
  }
  indicator = memcmp((const char*)topic, "/pillbox/noon", strlen("/pillbox/noon"));
  if (indicator == 0) {
    String s = String((char*)payload);
    preferences.putUInt("noon", atol(data));
    readConfig();
    mqtt_client.publish("/pillbox", "ok");
  }
  indicator = memcmp((const char*)topic, "/pillbox/night", strlen("/pillbox/night"));
  if (indicator == 0) {
    String s = String((char*)payload);
    preferences.putUInt("night", atol(data));
    readConfig();
    mqtt_client.publish("/pillbox", "ok");
  }

  indicator = memcmp((const char*)topic, "/pillbox/test/pillSuccess", strlen("/pillbox/test/pillSuccess"));
  if (indicator == 0) {
    String s = String((char*)payload);
    mqtt_client.publish("/pillbox/pillSuccess", data);
  }

  indicator = memcmp((const char*)topic, "/pillbox/test/pillFail", strlen("/pillbox/test/pillFail"));
  if (indicator == 0) {
    String s = String((char*)payload);
    mqtt_client.publish("/pillbox/pillFail", data);
  }


  indicator = memcmp((const char*)topic, "/pillbox/test/allTime", strlen("/pillbox/test/allTime"));
  if (indicator == 0) {
    showTime();
  }
  // int indicator = memcmp((const char*)payload, "takePill", 8);
  // if(indicator == 0)
  // {
  //   takePill();
  // }

  // // Switch on the LED if an 1 was received as first character
  // if ((char)payload[0] == '1') {
  //   digitalWrite(BUILTIN_LED, LOW);   // Turn the LED on (Note that LOW is the voltage level
  //   // but actually the LED is on; this is because
  //   // it is active low on the ESP-01)
  // } else {
  //   digitalWrite(BUILTIN_LED, HIGH);  // Turn the LED off by making the voltage HIGH
  // }
}

void mqtt_reconnect() {
  // Loop until we're reconnected
  while (!mqtt_client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "pillbox-client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (mqtt_client.connect(clientId.c_str(), "username", "password")) {
      Serial.println("connected");
      // Once connected, publish an announcement...
      // client.publish("outTopic", "hello world");
      // // ... and resubscribe
      mqtt_client.subscribe("/pillbox/RTC");
      mqtt_client.subscribe("/pillbox/morning");
      mqtt_client.subscribe("/pillbox/noon");
      mqtt_client.subscribe("/pillbox/night");
      mqtt_client.subscribe("/pillbox/test/pillFail");
      mqtt_client.subscribe("/pillbox/test/pillSuccess");
      mqtt_client.subscribe("/pillbox/test/allTime");
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqtt_client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}



void setup() {
  Serial.begin(115200);
  pinMode(IN_CABINET_SWITCH, INPUT_PULLUP);
  preferences.begin("nvs");
  readConfig();

  EasyBuzzer.setPin(23);
  if (!rtc.begin()) {
    Serial.println("Couldn't find RTC");
    Serial.flush();
  }

  if (rtc.lostPower()) {
    Serial.println("RTC lost power, let's set the time!");
    // When time needs to be set on a new device, or after a power loss, the
    // following line sets the RTC to the date & time this sketch was compiled
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    // This line sets the RTC with an explicit date & time, for example to set
    // January 21, 2014 at 3am you would call:
    //rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));
  }

  buttonPauseEx.attach(16, INPUT_PULLUP);
  buttonPauseEx.interval(25);
  buttonPauseEx.setPressedState(LOW);

  // stepper.setMaxSpeed(200);
  // stepper.setAcceleration(200);

  stepper.setMaxSpeed(1000.0);
  stepper.setAcceleration(200.0);
  stepper.setSpeed(200);

  // Begin WiFi section
  Serial.printf("\nConnecting to %s", ssid);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // print out info about the connection:
  Serial.println("\nConnected to network");
  Serial.print("My IP address is: ");
  Serial.println(WiFi.localIP());

  mqtt_client.setServer("broker.hivemq.com", 1883);
  mqtt_client.setCallback(mqtt_callback);

  runPillCheck();
}

void runPillCheck() {
  intervalCheckRunId = timerForBeep.setInterval([]() {
    DateTime now = rtc.now();
    uint32_t timestamp = now.unixtime();

    if (timestamp > 0 && isWaitTakePill == 0) {
      Serial.printf("p");
      if (timestamp + 1 >= eeMorning && eeMorning - timestamp < 4) {
        tempHandlerTime = String(eeMorning);
        timerForBeep.cancel(intervalCheckRunId);
        Serial.printf("早上吃藥觸發!");
        takePill();
      }

      if (timestamp + 1 >= eeNoon && eeNoon - timestamp < 4) {
        tempHandlerTime = String(eeNoon);
        timerForBeep.cancel(intervalCheckRunId);
        Serial.printf("ˊ中午吃藥觸發!");
        takePill();
      }

      if (timestamp + 1 >= eeNight && eeNight - timestamp < 4) {
        tempHandlerTime = String(eeNight);
        timerForBeep.cancel(intervalCheckRunId);
        Serial.printf("晚上吃藥觸發!");
        takePill();
      } else {
        // Serial.printf(" no.. ");
      }
    }
    else
    {
          Serial.printf("c");
    }
  },
                                                1000);
}

String convertStr(int oo) {
  if (oo <= 9) {
    return String('0') + String(oo);
  }
  return String(oo);
}

String getTime() {
  DateTime now = rtc.now();
  String timestamp = now.timestamp(DateTime::TIMESTAMP_FULL);
  String str = "";
  str += now.year();
  str += '/';
  str += convertStr(now.month());
  str += '/';
  str += convertStr(now.day());
  str += ' ';
  str += convertStr(now.hour());
  str += ':';
  str += convertStr(now.minute());
  str += ':';
  str += convertStr(now.second());
  str += ' ';
  str += timestamp;
  return str;
}


void showTime() {
  DateTime now = rtc.now();
  uint32_t timestamp = now.unixtime();
  Serial.print(now.year(), DEC);
  Serial.print('/');
  Serial.print(now.month(), DEC);
  Serial.print('/');
  Serial.print(now.day(), DEC);
  Serial.print(" (");
  Serial.print(daysOfTheWeek[now.dayOfTheWeek()]);
  Serial.print(") ");
  Serial.print(now.hour(), DEC);
  Serial.print(':');
  Serial.print(now.minute(), DEC);
  Serial.print(':');
  Serial.print(now.second(), DEC);
  Serial.print(' ');

  Serial.print(timestamp);
  Serial.print(' ');
  Serial.print(eeMorning);
  Serial.print(' ');
  Serial.print(eeNoon);
  Serial.print(' ');
  Serial.print(eeNight);

  Serial.println(' ');
}

void inerstDB() {
}

void takePill() {
  if (isWaitTakePill == 1) return;
  isWaitTakePill = 1;
  stepper.setCurrentPosition(0);
  stepper.moveTo(512);
  showTime();

  cabinetStatus = digitalRead(IN_CABINET_SWITCH);
  Serial.print(cabinetStatus, DEC);
  Serial.println();
  timerForBeep.cancelAll();
  timerForBeep.setInterval([]() {
    Serial.println("暫時未取藥 !");
    EasyBuzzer.singleBeep(
      1500,
      BEEP_DURATION);
  },
                           BEEP_ITERVAL);

  timerForBeep.setInterval([]() {
    inerstDB();
  },
                           2000);

  timerForBeep.setInterval([]() {
    cabinetStatus = digitalRead(IN_CABINET_SWITCH);
    if (cabinetStatus == 1) {
      mqtt_client.publish("/pillbox/pillSuccess", tempHandlerTime.c_str());
      isWaitTakePill = 0;
      Serial.println("已取藥 !");
      timerForBeep.cancelAll();
      runPillCheck();
      return;
    }
  },
                           1000);

  timerForBeep.setTimeout([]() {
    cabinetStatus = digitalRead(IN_CABINET_SWITCH);
    if (cabinetStatus == 0) {
      mqtt_client.publish("/pillbox/pillFail", tempHandlerTime.c_str());
      isWaitTakePill = 0;
      Serial.println("確定未取藥 !");
      timerForBeep.cancelAll();
      runPillCheck();
    }
  },
                          PICKER_PILL_FAIL_DURATION);
}


void loop() {




  if (buttonPauseEx.pressed() && isWaitTakePill == 0) {
    takePill();
  }


  if (!mqtt_client.connected()) {
    mqtt_reconnect();
  }

  mqtt_client.loop();
  stepper.run();
  buttonPauseEx.update();
  EasyBuzzer.update();
  timerForBeep.handle();
}
