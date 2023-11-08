import React, { useState, useEffect, useRef } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
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

  const startRecordingSensorData = () => {
    intervalIdRef.current = []
    intervalIdRef.current = setInterval(() => {
      let currentJsonData = jsonData.current
      currentJsonData.push({
        "AccelX": accelData.x.toString(),
        "AccelY": accelData.y.toString(),
        "AccelZ": accelData.z.toString(),
        "GyroX": gyroData.x.toString(),
        "GyroY": gyroData.y.toString(),
        "GyroZ": gyroData.z.toString()
      })
    }, 50)
  }

  const stopRecordingSensorData = () => {
    clearInterval(intervalIdRef.current)
    saveDataAsCSV(jsonData.current)
  }

  useEffect(() => {
    _subscribe()
    Accelerometer.setUpdateInterval(30);
    Gyroscope.setUpdateInterval(30);
    return () => _unsubscribe();
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
