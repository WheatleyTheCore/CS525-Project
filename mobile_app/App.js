import React, { useState, useEffect, useRef } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View, Image } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { Audio } from 'expo-av';
import { Picker } from '@react-native-picker/picker';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native'
import { saveDataAsCSV } from './utils/saveDataAsCSV';
import { MNISTDataset } from 'tfjs-data-mnist';
import { MaterialIcons } from '@expo/vector-icons';

import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer, DefaultTheme } from "@react-navigation/native";
import RecordScreen from './screens/recordScreen';
import RecognizeScreen from './screens/recognizeScreen';

const modelJSON = require('./prodModel/model.json')
const modelWeights = require('./prodModel/group1-shard1of1.bin')

const Tab = createBottomTabNavigator();
const theme_color = '#B71C1C'

export default function App() {

    return (
        <NavigationContainer theme={MyTheme}>
            <Tab.Navigator
                screenOptions={({ route }) => ({
                    headerStyle: {
                        backgroundColor: theme_color,
                    },
                    headerTintColor: '#ffffff',
                    tabBarStyle: {
                        backgroundColor: '#ffffff',
                    },
                    tabBarActiveTintColor: theme_color,
                    tabBarIcon: ({ color, size }) => {
                        let iconName;

                        if (route.name === 'Detect') {
                            iconName = 'search'; // Change this to the name of your icon for 'Detect'
                        } else if (route.name === 'Record') {
                            iconName = 'input'; // Change this to the name of your icon for 'Record'
                        }

                        return <MaterialIcons name={iconName} size={size} color={color} />;
                    },
                })}
            >
                <Tab.Screen name="Detect" component={RecognizeScreen} />
                <Tab.Screen name="Record" component={RecordScreen} />
            </Tab.Navigator>
        </NavigationContainer>
    );
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

