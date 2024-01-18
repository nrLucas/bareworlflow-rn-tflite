import React from "react";
import { Alert, Text, View } from "react-native";
import { scanOCR } from "vision-camera-ocr";
import { runOnJS } from "react-native-reanimated";
import { Camera, useCameraDevice, useFrameProcessor, CameraPosition } from "react-native-vision-camera";

import { Loading } from "../../components/Loading";
import { styles } from "./styles";
import { Button } from "../../components/Button";

export function Home() {
    const [text, setText] = React.useState("");
    const [cameraPosition, setCameraPosition] = React.useState("front");
    const [isFlashOn, setIsFlashOn] = React.useState(false);

    const devices = useCameraDevice();
    const deviceCP = cameraPosition == "front" ? devices.front : devices.back;

    const frameProcessor = useFrameProcessor((frame) => {
        "worklet";
        const data = scanOCR(frame);

        runOnJS(setText)(data.result.text);
    }, []);
}
