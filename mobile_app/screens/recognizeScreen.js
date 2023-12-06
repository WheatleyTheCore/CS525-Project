import React, { useState, useEffect, useRef, useId } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View, Image } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { Audio } from 'expo-av';
import { Picker } from '@react-native-picker/picker';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native'
import { DefaultTheme } from "@react-navigation/native";
const theme_color = '#B71C1C'

const modelJSON = require('../prodModel/model.json')
const modelWeights = require('../prodModel/group1-shard1of1.bin')

const output_decoder = {
    0: 'Sitting',
    1: 'Standing',
    2: "Walking",
    3: "Crouching"
}

const audioPaths = {
    sit: require('../assets/sitting.mp3'),
    stand: require('../assets/standing.mp3'),
    walk: require('../assets/walking.mp3'),
    crouch: require('../assets/crouching.mp3'),
};

const imgPaths = {
    start: require('../assets/start.png'),
    sit: require('../assets/sit.png'),
    stand: require('../assets/stand.png'),
    walk: require('../assets/walk.png'),
    crouch: require('../assets/crouch.png'),
};

function RecognizeScreen() {

    const [isTfReady, setIsTfReady] = useState(false)
    const [model, setModel] = useState()
    const [ds, setDs] = useState()

    const previousPrediction = useRef(null)
    const [prediction, setPrediction] = useState("none")


    const bufferRef = useRef(new Array(40).fill([0,0,0,0,0,0]))
    const bufferHead = useRef(0)

    const [crouchSound, setCrouchSound] = useState()
    const [standSound, setStandSound] = useState()
    const [sitSound, setSitSound] = useState()
    const [walkSound, setWalkSound] = useState()

    const addToBuffer = (item) => {                                  // sorta not so great circular buffer
        let updatedArray = [...bufferRef.current]
        updatedArray[bufferHead.current] = item
        //console.log(updatedArray)
        bufferRef.current = [...updatedArray]
        bufferHead.current = (bufferHead.current + 1) % 40
    }

    const getBufferContent = () => {
        let bufferArray = []
        for (let i = 0; i < 40; i++) {
            bufferArray.push(bufferRef.current[(bufferHead.current + i) % 40])
        }
        return bufferArray
    }

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

    const recordIntervalIdRef = useRef(null)
    const inferIntervalIdRef = useRef(null)

    const _subscribe = () => {
        setAccelSubscription(Accelerometer.addListener(data => setAccelData(data)));
        setGyroSubscription(Gyroscope.addListener(data => setGyroData(data)))
    };

    const _unsubscribe = () => {
        accelSubscription && accelSubscription.remove()
        gyroSubscription && gyroSubscription.remove()
        recordIntervalIdRef && clearInterval(recordIntervalIdRef)
        inferIntervalIdRef && clearInterval(inferIntervalIdRef)
        setAccelSubscription(null)
        setGyroSubscription(null)
    };

    const startRecordingSensorData = async () => {
        recordIntervalIdRef.current = setInterval(() => {
            setAccelData(accelData => { // using the set functions to get state within the setinterval is hacky and very bad
                setGyroData(gyroData => { // this is horrible for performance and my sanity
                    addToBuffer([accelData.x, accelData.y, accelData.z, gyroData.x, gyroData.y, gyroData.z])
                    return gyroData
                })
                return accelData
            })
        }, 50)
    }

    const delay = (delayInms) => {
        return new Promise(resolve => setTimeout(resolve, delayInms));
    };

    useEffect(() => {
        const setUpTfAndRecording = async () => {
            await tf.ready()
            const model = await tf.loadGraphModel(
                bundleResourceIO(modelJSON, modelWeights)
            ).catch((e) => {
                console.log("[LOADING ERROR] info:", e)
            })
            setModel(model)
            await tf.ready()
            setIsTfReady(true)
            _subscribe()
            await (delay(50))
            startRecordingSensorData()
            inferIntervalIdRef.current = setInterval(async () => {
                let rawOutput = await model.predict(tf.tensor(getBufferContent()).expandDims().reshape([-1, 40, 6, 1])).data()
                let outputClass = output_decoder[Object.values(rawOutput).indexOf(Math.max(...Object.values(rawOutput)))]
                if (outputClass != previousPrediction.current) {
                    previousPrediction.current = outputClass
                    setPrediction(outputClass)
                }
        
            }, 1000)
        }

        const setUpSounds = async () => {
            console.log('Loading Sounds');
            const standSound = new Audio.Sound();
            const sitSound = new Audio.Sound();
            const crouchSound = new Audio.Sound();
            const walkSound = new Audio.Sound();
            await standSound.loadAsync(require('../assets/standing.mp3'), { shouldPlay: true })
            await sitSound.loadAsync(require('../assets/sitting.mp3'), { shouldPlay: true })
            await crouchSound.loadAsync(require('../assets/crouching.mp3'), { shouldPlay: true })
            await walkSound.loadAsync(require('../assets/walking.mp3'), { shouldPlay: true })
            setStandSound(standSound)
            setWalkSound(walkSound)
            setCrouchSound(crouchSound)
            setSitSound(sitSound)
        }

        setUpTfAndRecording()
        setUpSounds()

        Audio.setAudioModeAsync({
            playsInSilentModeIOS: true,
        })


        return () => _unsubscribe()
    }, [])

    useEffect(() => {
        handlePrediction(prediction)
    }, [prediction])
    const [buttonImage, setButtonImage] = useState(imgPaths.start);

    const handlePrediction = async (pred) => {

        //Change the image source based on the prediction
        switch (pred) {
            case 'Sitting':
                setButtonImage(imgPaths.sit);

                await sitSound.setPositionAsync(0);
                await sitSound.playAsync()

                break;

            case 'Standing':
                setButtonImage(imgPaths.stand);

                await standSound.setPositionAsync(0);
                await standSound.playAsync()

                break;

            case 'Walking':
                setButtonImage(imgPaths.walk);
                
                await walkSound.setPositionAsync(0);
                await walkSound.playAsync()

                break;

            case 'Crouching':
                setButtonImage(imgPaths.crouch);
                
                await crouchSound.setPositionAsync(0);
                await crouchSound.playAsync()

                break;

            default:
                setButtonImage(imgPaths.start);
        }
    };

    return (

        <View style={img_styles.container}>

            <TouchableOpacity>
                <Image source={buttonImage} style={img_styles.imageButton} />
            </TouchableOpacity>
            <Text>{prediction}</Text>

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

export default RecognizeScreen