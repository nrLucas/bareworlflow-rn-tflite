// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require("expo/metro-config");

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Adicionando ou atualizando as extensões de arquivo
config.resolver.assetExts = [...(config.resolver.assetExts || []), "tflite", "bin", "png", "jpg", "txt"];

module.exports = config;
