# CS 525 Final Project

This is our final project for CS 525 (On-Device Deep Learning) at WPI.

## Overview
This smartphone app is intended to figure out what position/action the user is doing while it is in their pocket (e.g. walking, standing, sitting, climbing stairs, etc) using a TensorFlow model running on the device.

## Development Setup
### Mobile App
1. cd into the mobile_app directory
2. Run `npm i` to install all necessary npm packages
3. Test using `npx expo start --tunnel`

### ML
1. cd into ml directory
2. (optional) start whatever python virtual environment you like to use if you like to use them
3. run pip (pip3 if you have python3) install -r requirements.txt **TODO: This will not complete, since we have not added requirements.txt because we do not know what we need quite yet** 

## Development Notes

1. Expo is kinda finicky. To get it to actually connect, you want to run the expo server in tunnel mode using `npx expo start --tunnel`.
2. https://github.com/tensorflow/tfjs-examples/tree/master/react-native are some great examples of using tensorflow on mobile devices. 
3. https://www.tensorflow.org/js/tutorials/applications/react_native -- The docs for tensorflow react native
4. https://www.tensorflow.org/lite/guide -- the docs for tf lite, which I believe tf-react uses under the hood


## Stretch Goals
1. also be able to classify things that happen over time, lke walking. Minimum viable product is something that can classify static positions.
2. Maybe implement little app that will notice when you've been sitting for over an hour and tell you it's time to stand up and move around a bit.
