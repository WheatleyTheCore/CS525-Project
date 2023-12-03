import React, { useState, useEffect, useRef } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View, Image } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { Audio } from 'expo-av';
import { Picker } from '@react-native-picker/picker';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native'
import { saveDataAsCSV } from '../utils/saveDataAsCSV';
import { MNISTDataset } from 'tfjs-data-mnist';
import { MaterialIcons } from '@expo/vector-icons';
import { DefaultTheme } from "@react-navigation/native";

const theme_color = '#B71C1C'
const RecordScreen = () => {
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
        await startSound.loadAsync(require('../assets/mariostart.mp3'), { shouldPlay: true })
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



    return (
        <View style={styles.container}>

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
            <Button
                title="Start Recording"
                color={theme_color}
                onPress={() => startRecordingSensorData()}
            />
            <Button
                title="Stop Recording"
                color={theme_color}
                onPress={() => stopRecordingSensorData()}
            />

        </View>
    )
}

const MyTheme = {
    ...DefaultTheme,
    colors: {
        ...DefaultTheme.colors,
        background: '#ffffff',
        text: '#000000'
    },
};

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

const img_styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    imageButton: {
        width: 300,
        height: 300,
        // Add any additional styling you need for your image button
    },
});

export default RecordScreen