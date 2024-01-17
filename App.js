import React, { useEffect, useState, useRef } from "react";
import { StyleSheet, Text, View, SafeAreaView, Button, Image } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";

import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system";
// import * as jpeg from "jpeg-js";

import * as MediaLibrary from "expo-media-library";

import { Camera } from "expo-camera";
import { StatusBar } from "expo-status-bar";

//import TensorFlowLite from 'react-native-tensorflow-lite';

// const loadModel = async () => {
//     //.ts: const loadModel = async ():Promise<void|tf.LayersModel>=>{
//     const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights)).catch((e) => {
//         console.log("[LOADING ERROR] info:", e);
//     });
//     return model;
// };

const loadModel = async () => {
    const modelJSON = require("./assets/layers/model.json");
    const modelWeights1 = require("./assets/layers/group1-shard1of4.bin");
    const modelWeights2 = require("./assets/layers/group1-shard2of4.bin");
    const modelWeights3 = require("./assets/layers/group1-shard3of4.bin");
    const modelWeights4 = require("./assets/layers/group1-shard4of4.bin");

    if (modelJSON && modelWeights1 && modelWeights2 && modelWeights3 && modelWeights4) {
        console.log("Carregados");

        try {
            const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, [modelWeights1, modelWeights2, modelWeights3, modelWeights4]));
            console.log("Model loaded successfully");
            return model;
        } catch (e) {
            console.error("Error loading model", e);
        }
    }
};

const transformImageToTensor = async (uri) => {
    //.ts: const transformImageToTensor = async (uri:string):Promise<tf.Tensor>=>{
    //read the image as base64
    // const img64 = await FileSystem.readAsStringAsync(uri, {
    //     encoding: FileSystem.EncodingType.Base64,
    // });
    const imgBuffer = tf.util.encodeString(uri, "base64").buffer;
    const raw = new Uint8Array(imgBuffer);
    let imgTensor = decodeJpeg(raw);
    const scalar = tf.scalar(255);
    //resize the image
    imgTensor = tf.image.resizeNearestNeighbor(imgTensor, [300, 300]);
    //normalize; if a normalization layer is in the model, this step can be skipped
    const tensorScaled = imgTensor.div(scalar);
    //final shape of the rensor
    const img = tf.reshape(tensorScaled, [1, 300, 300, 3]);
    return img;
};

const makePredictions = async (batch, model, imagesTensor) => {
    //.ts: const makePredictions = async (batch:number, model:tf.LayersModel,imagesTensor:tf.Tensor<tf.Rank>):Promise<tf.Tensor<tf.Rank>[]>=>{
    //cast output prediction to tensor
    const predictionsdata = model.predict(imagesTensor);
    //.ts: const predictionsdata:tf.Tensor = model.predict(imagesTensor) as tf.Tensor
    let pred = predictionsdata.split(batch); //split by batch size
    //return predictions
    return pred;
};

export const getPredictions = async (image) => {
    await tf.ready();
    const model = await loadModel();
    if (model) console.log("model", model);
    const tensor_image = await transformImageToTensor(image);

    if (tensor_image) console.log("tensor_image", tensor_image);

    if (model && tensor_image) {
        const predictions = await makePredictions(1, model, tensor_image);

        if (predictions) console.log("predictions", predictions);
        return predictions;
    } else {
        console.log("Não deu");
        return false;
    }
};

const App = () => {
    const [isTfReady, setIsTfReady] = useState(false);
    const [result, setResult] = useState("");
    const [pickedImage, setPickedImage] = useState(false);
    const [loading, setLoading] = useState(false);
    const [photo, setPhoto] = useState();

    const pickImage = async () => {
        // No permissions request is necessary for launching the image library
        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.All,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });
        if (!result.cancelled) {
            console.log("result uri", result.uri);
            setPickedImage(result.uri);
        }
    };

    useEffect(() => {
        if (!!pickedImage) {
            setLoading(true);
            // classifyUsingMobilenet(pickedImage);
        }
    }, [pickedImage]);

    useEffect(() => {
        console.log("aqui");
        if (!!photo) {
            setLoading(true);

            const aux = getPredictions(photo.base64);
            if (!!aux) console.log("aux", aux);
            // classifyUsingMobilenet(photo.base64);
        }
    }, [photo]);

    //CAMERA
    let cameraRef = useRef();
    const [hasCameraPermission, setHasCameraPermission] = useState();
    const [hasMediaLibraryPermission, setHasMediaLibraryPermission] = useState();

    useEffect(() => {
        (async () => {
            const cameraPermission = await Camera.requestCameraPermissionsAsync();
            const mediaLibraryPermission = await MediaLibrary.requestPermissionsAsync();
            setHasCameraPermission(cameraPermission.status === "granted");
            setHasMediaLibraryPermission(mediaLibraryPermission.status === "granted");
        })();
    }, []);

    if (hasCameraPermission === undefined) {
        return <Text>Requesting permissions...</Text>;
    } else if (!hasCameraPermission) {
        return <Text>Permission for camera not granted. Please change this in settings.</Text>;
    }

    let takePic = async () => {
        let options = {
            quality: 1,
            base64: true,
            exif: false,
        };

        let newPhoto = await cameraRef.current.takePictureAsync(options);
        setPhoto(newPhoto);
    };

    if (photo) {
        // let sharePic = () => {
        //   shareAsync(photo.uri).then(() => {
        //     setPhoto(undefined);
        //   });
        // };

        // let savePhoto = () => {
        //   MediaLibrary.saveToLibraryAsync(photo.uri).then(() => {
        //     setPhoto(undefined);
        //   });
        // };

        return (
            <SafeAreaView style={styles.container}>
                <Image style={styles.preview} source={{ uri: "data:image/jpg;base64," + photo.base64 }} />
                {/* <View style={styles.buttonContainer}>
          <Button title="Take Pic" onPress={takePic} />
        </View> */}
                {loading ? <Text>Carregando...</Text> : result !== "" ? <Text>{result}</Text> : <Text>Modelo pronto!</Text>}
                {/* <Button title="Share" onPress={sharePic} />
        {hasMediaLibraryPermission ? (
          <Button title="Save" onPress={savePhoto} />
        ) : undefined}*/}
                <Button title="Voltar" onPress={() => setPhoto(undefined)} />
            </SafeAreaView>
        );
    }

    return (
        <Camera style={styles.container} ref={cameraRef}>
            <View
                style={{
                    height: "100%",
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end",
                }}
            >
                {!!pickedImage && <Image source={{ uri: pickedImage }} style={{ width: 200, height: 200, margin: 40 }} />}

                <View
                    style={{
                        backgroundColor: "#ffff",
                        width: "100%",
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                    }}
                >
                    {loading ? <Text>Carregando...</Text> : result !== "" ? <Text>{result}</Text> : <Text>Modelo pronto!</Text>}

                    <View
                        style={{
                            display: "flex",
                            flexDirection: "row",
                            justifyContent: "space-around",
                            width: "100%",
                            paddingBottom: "5%",
                        }}
                    >
                        <Button title="Galeria" onPress={pickImage} />
                        <Button title="Tirar Foto" onPress={takePic} />
                    </View>
                </View>

                {/* <View style={{ width: "100%", height: 20 }} /> */}
                {/* {!isTfReady && <Text>Loading TFJS model...</Text>} */}
            </View>
        </Camera>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
    },
    buttonContainer: {
        backgroundColor: "#fff",
        alignSelf: "flex-end",
    },
    textContainer: {
        background: "#ffff",
    },
    preview: {
        alignSelf: "stretch",
        flex: 1,
    },
});

export default App;

