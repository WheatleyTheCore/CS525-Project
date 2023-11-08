import React, { useState, useEffect, useRef } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { Audio } from 'expo-av';
import { saveDataAsCSV } from './utils/saveDataAsCSV';

export default function App() {
  const [accelData, setAccelData] = useState({
    x: 0,
    y: 0,
    z: 0,
  });

  const [gyroData, setGyroData] = useState({
    x: 0,
    y: 0,
    z: 0,
  });

  const [accelSubscription, setAccelSubscription] = useState(null);
  const [gyroSubscription, setGyroSubscription] = useState(null);

  const [startSound, setStartSound] = useState()
  const [endSound, setEndSound] = useState()


  // id for setInterval that we use for recording sensor data.
  const intervalIdRef = useRef(null)

  let jsonData = useRef([])

  const _subscribe = () => {
    setAccelSubscription(Accelerometer.addListener(data => setAccelData(data)));
    setGyroSubscription(Gyroscope.addListener(data => setGyroData(data)))
  };

  const _unsubscribe = () => {
    accelSubscription && accelSubscription.remove()
    gyroSubscription && gyroSubscription.remove()
    setAccelSubscription(null)
    setGyroSubscription(null)
  };

  const delay = (delayInms) => {
    return new Promise(resolve => setTimeout(resolve, delayInms));
  };

  const startRecordingSensorData = async () => {

    jsonData.current = []

    // load sounds
    console.log('Loading Sound');
    const startSound = new Audio.Sound();
    await startSound.loadAsync(require('./assets/mariostart.mp3'), {shouldPlay: true})
    setStartSound(startSound)

    console.log('Playing Sound');
    await startSound.setPositionAsync(0);
    await startSound.playAsync()

    await delay(5000) // wait 5 seconds for the sound to finish playing. Definitely a better way to do this but I don't have time to find it atm

    console.log('recording!')
    intervalIdRef.current = setInterval(() => {
      let updatedJsonData = jsonData.current
      setAccelData(accelData => { // using the set functions to get state within the setinterval is hacky and very bad
        setGyroData(gyroData => { // this is horrible for performance and my sanity
          updatedJsonData.push({
            "AccelX": accelData.x.toString(),
            "AccelY": accelData.y.toString(),
            "AccelZ": accelData.z.toString(),
            "GyroX": gyroData.x.toString(),
            "GyroY": gyroData.y.toString(),
            "GyroZ": gyroData.z.toString()
          })
          return gyroData
        })
        return accelData
      })
      jsonData.current = updatedJsonData
    }, 50)
  }

  const stopRecordingSensorData = () => {
    if (intervalIdRef.current) {
      clearInterval(intervalIdRef.current)
      intervalIdRef.current = []
      let processedJsonData = [...jsonData.current]
      //console.log(Array.isArray(processedJsonData))
      jsonData.current = []
      for (let i = 0; i < 100; i++) { //remove stuff from last 5 sconds (i.e. last 100 elements since we record every 50 ms). also kinda hacky
        processedJsonData.pop()
      }
      saveDataAsCSV(processedJsonData)
    }
  }

  useEffect(() => {
    _subscribe()
    Accelerometer.setUpdateInterval(30);
    Gyroscope.setUpdateInterval(30);
    Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,
  })
    return () => {
      _unsubscribe();
      if (startSound) startSound.unloadAsync();
      if (endSound) endSound.unloadAsync();
    }
  }, []);


  return (
    <View style={styles.container}>
      <Text style={styles.text}>accelX: {accelData.x.toFixed(2)}</Text>
      <Text style={styles.text}>accelY: {accelData.y.toFixed(2)}</Text>
      <Text style={styles.text}>accelZ: {accelData.z.toFixed(2)}</Text>
      {/* <View style={styles.buttonContainer}>
        <TouchableOpacity onPress={subscription ? _unsubscribe : _subscribe} style={styles.button}>
          <Text>{subscription ? 'On' : 'Off'}</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={_slow} style={[styles.button, styles.middleButton]}>
          <Text>Slow</Text>
        </TouchableOpacity>
        
      </View> */}
      <Button title="Start Recording" onPress={() => startRecordingSensorData()} />
      <Button title="Stop Recording" onPress={() => stopRecordingSensorData()} />

    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  text: {
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    alignItems: 'stretch',
    marginTop: 15,
  },
  button: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#eee',
    padding: 10,
  },
  middleButton: {
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: '#ccc',
  },
});
