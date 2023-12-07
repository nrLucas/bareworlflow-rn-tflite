import React from "react";
import { StyleSheet, Text, View, SafeAreaView, Button, Image } from "react-native";
import { StatusBar } from "expo-status-bar";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";
import * as ImagePicker from "expo-image-picker";
import * as MediaLibrary from "expo-media-library";

import { Camera } from "expo-camera";

const modelJSON = require("./assets/model.json");
const modelWeights = require("./assets/group1_shard1.bin");

const loadModel = async () => {
    //.ts: const loadModel = async ():Promise<void|tf.LayersModel>=>{
    const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights)).catch((e) => {
        console.log("[LOADING ERROR] info:", e);
    });
    return model;
};

const transformImageToTensor = async (uri) => {
    //.ts: const transformImageToTensor = async (uri:string):Promise<tf.Tensor>=>{
    //read the image as base64
    // const img64 = await FileSystem.readAsStringAsync(uri, {
    //     encoding: FileSystem.EncodingType.Base64,
    // });
    // const imgBuffer = tf.util.encodeString(img64, "base64").buffer;
    const raw = new Uint8Array(uri);
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
    const tensor_image = await transformImageToTensor(image);
    const predictions = await makePredictions(1, model, tensor_image);
    return predictions;
};

export default function App() {
    const [pickedImage, setPickedImage] = React.useState(false);
    const [loading, setLoading] = React.useState(true);
    const [hasCameraPermission, setHasCameraPermission] = React.useState();
    const [hasMediaLibraryPermission, setHasMediaLibraryPermission] = React.useState();
    const [photo, setPhoto] = React.useState();
    const [result, setResult] = React.useState("");

    const cameraRef = React.useRef();

    React.useEffect(() => {
        (async () => {
            const cameraPermission = await Camera.requestCameraPermissionsAsync();
            const mediaLibraryPermission = await MediaLibrary.requestPermissionsAsync();
            setHasCameraPermission(cameraPermission.status === "granted");
            setHasMediaLibraryPermission(mediaLibraryPermission.status === "granted");
        })();
    }, []);

    React.useEffect(() => {
        if (!!photo) {
            console.log("photo ========>", photo.uri);
            console.log("============================");
            setLoading(true);
            setResult("");
            const result = getPredictions(photo.base64);

            if (!!result) {
                console.log("result", result);
                setResult(result);
                setLoading(false);
            }
        }
    }, [photo]);

    if (hasCameraPermission === undefined) {
        return <Text>Requesting permissions...</Text>;
    } else if (!hasCameraPermission) {
        return <Text>Permission for camera not granted. Please change this in settings.</Text>;
    }

    const takePick = async () => {
        let options = {
            quality: 1,
            base64: true,
            exif: false,
        };

        let newPhoto = await cameraRef.current.takePictureAsync(options);
        setPhoto(newPhoto);
    };

    if (photo) {
        return (
            <SafeAreaView style={styles.container}>
                <Image style={styles.preview} source={{ uri: "data:image/jpg;base64," + photo.base64 }} />

                {loading ? <Text>Carregando...</Text> : result !== "" ? <Text>{result}</Text> : <Text>Modelo pronto!</Text>}

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
                        {/* <Button title="Galeria" onPress={pickImage} /> */}
                        <Button title="Tirar Foto" onPress={() => takePick()} />
                    </View>
                </View>

                {/* <View style={{ width: "100%", height: 20 }} /> */}
                {/* {!isTfReady && <Text>Loading TFJS model...</Text>} */}
            </View>
        </Camera>
    );
}

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
