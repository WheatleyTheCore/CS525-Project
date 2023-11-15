import React, { useState, useEffect, useRef } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { Audio } from 'expo-av';
import { Picker } from '@react-native-picker/picker';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native'
import { saveDataAsCSV } from './utils/saveDataAsCSV';
import { MNISTDataset } from 'tfjs-data-mnist';


const modelJSON = require('./model/model.json')
const modelWeights = require('./model/group1-shard1of1.bin')

const MNISTnumber = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764705882352941, 0.07058823529411765, 0.07058823529411765, 0.07058823529411765, 0.49411764705882355, 0.5333333333333333, 0.6862745098039216, 0.10196078431372549, 0.6509803921568628, 1.0, 0.9686274509803922, 0.4980392156862745, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11764705882352941, 0.1411764705882353, 0.3686274509803922, 0.6039215686274509, 0.6666666666666666, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.8823529411764706, 0.6745098039215687, 0.9921568627450981, 0.9490196078431372, 0.7647058823529411, 0.25098039215686274, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19215686274509805, 0.9333333333333333, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.984313725490196, 0.36470588235294116, 0.3215686274509804, 0.3215686274509804, 0.2196078431372549, 0.15294117647058825, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07058823529411765, 0.8588235294117647, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.7764705882352941, 0.7137254901960784, 0.9686274509803922, 0.9450980392156862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3137254901960784, 0.611764705882353, 0.4196078431372549, 0.9921568627450981, 0.9921568627450981, 0.803921568627451, 0.043137254901960784, 0.0, 0.16862745098039217, 0.6039215686274509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.054901960784313725, 0.00392156862745098, 0.6039215686274509, 0.9921568627450981, 0.35294117647058826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5450980392156862, 0.9921568627450981, 0.7450980392156863, 0.00784313725490196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043137254901960784, 0.7450980392156863, 0.9921568627450981, 0.27450980392156865, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13725490196078433, 0.9450980392156862, 0.8823529411764706, 0.6274509803921569, 0.4235294117647059, 0.00392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3176470588235294, 0.9411764705882353, 0.9921568627450981, 0.9921568627450981, 0.4666666666666667, 0.09803921568627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17647058823529413, 0.7294117647058823, 0.9921568627450981, 0.9921568627450981, 0.5882352941176471, 0.10588235294117647, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06274509803921569, 0.36470588235294116, 0.9882352941176471, 0.9921568627450981, 0.7333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9764705882352941, 0.9921568627450981, 0.9764705882352941, 0.25098039215686274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1803921568627451, 0.5098039215686274, 0.7176470588235294, 0.9921568627450981, 0.9921568627450981, 0.8117647058823529, 0.00784313725490196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15294117647058825, 0.5803921568627451, 0.8980392156862745, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9803921568627451, 0.7137254901960784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09411764705882353, 0.4470588235294118, 0.8666666666666667, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.788235294117647, 0.3058823529411765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09019607843137255, 0.25882352941176473, 0.8352941176470589, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.7764705882352941, 0.3176470588235294, 0.00784313725490196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07058823529411765, 0.6705882352941176, 0.8588235294117647, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.7647058823529411, 0.3137254901960784, 0.03529411764705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.21568627450980393, 0.6745098039215687, 0.8862745098039215, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.9568627450980393, 0.5215686274509804, 0.043137254901960784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5333333333333333, 0.9921568627450981, 0.9921568627450981, 0.9921568627450981, 0.8313725490196079, 0.5294117647058824, 0.5176470588235295, 0.06274509803921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

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

  const [label, setLabel] = useState("Sitting")

  const [isTfReady, setIsTfReady] = useState(false)
  const [model, setModel] = useState()
  const [ds, setDs] = useState()
  const [prediction, setPrediction] = useState("none")


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
    // handle intermediate re-record starts so we can have multiple things in one csv
    if (intervalIdRef.current) {
      clearInterval(intervalIdRef.current)
      let processedJsonData = [...jsonData.current]
      for (let i = 0; i < 120; i++) { //remove stuff from last 6 sconds (i.e. last 100 elements since we record every 50 ms). also kinda hacky
        processedJsonData.pop()
      }
      jsonData.current = processedJsonData
    }


    // load sounds
    console.log('Loading Sound');
    const startSound = new Audio.Sound();
    await startSound.loadAsync(require('./assets/mariostart.mp3'), { shouldPlay: true })
    setStartSound(startSound)

    console.log('Playing Sound');
    await startSound.setPositionAsync(0);
    await startSound.playAsync()

    await delay(5000) // wait 5 seconds for the sound to finish playing. Definitely a better way to do this but I don't have time to find it atm
    let dataLabel = label
    console.log(`recording ${dataLabel}!`)
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
            "GyroZ": gyroData.z.toString(),
            "Label": dataLabel
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
      intervalIdRef.current = null
      let processedJsonData = [...jsonData.current]
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


  useEffect(() => {
    const setUpTf = async () => {
      await tf.ready()
      const model = await tf.loadGraphModel(
        bundleResourceIO(modelJSON, modelWeights)
      ).catch((e) => {
        console.log("[LOADING ERROR] info:", e)
      })
      setModel(model)
      await tf.ready()
      setIsTfReady(true)
    }
    setUpTf()
  }, [])

  // useEffect(() => {
  //   const loadDataset = async () => {
  //     const ds = await MNISTDataset.create()
  //     console.log(ds)
  //     //setDs(ds.testDataset.iterator())
  //   }

  //   loadDataset()
  // }, [isTfReady])

  return (
    <View style={styles.container}>
      <Text>Tf: {isTfReady? "ready" : "not ready"}</Text>
      {isTfReady ? (
        <>
          <Button title="make prediction" onPress={() => setPrediction(JSON.stringify(model.predict(MNISTnumber[0])))} />
          <Text>{prediction}</Text>
        </>
      ) : null}
      <Text style={styles.text}>accelX: {accelData.x.toFixed(2)}</Text>
      <Text style={styles.text}>accelY: {accelData.y.toFixed(2)}</Text>
      <Text style={styles.text}>accelZ: {accelData.z.toFixed(2)}</Text>
      <Picker
        selectedValue={label}
        onValueChange={(val, index) => {
          console.log(val)
          setLabel(val)
        }}
      >
        <Picker.Item label="Sitting" value="Sitting" />
        <Picker.Item label="Standing" value="Standing" />
        <Picker.Item label="Walking" value="Walking" />
        <Picker.Item label="Crouching" value="Crouching" />
      </Picker>
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
