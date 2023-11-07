import { jsonToCSV, readRemoteFile } from 'react-native-csv';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import * as Sharing from "expo-sharing";

export async function makeCSV() {

    const jsonData = `[
    {
        "Column 1": "Name",
        "Column 2": "Surname",
        "Column 3": "Email",
        "Column 4": "Info"
    }
  ]`;

    const CSV = jsonToCSV(jsonData);

    // Name the File
    const directoryUri = FileSystem.documentDirectory;
    const fileUri = directoryUri + `formData.csv`;

    // Write the file to system
    FileSystem.writeAsStringAsync(fileUri, CSV)

    try {
        const UTI = 'public.item';
        await Sharing.shareAsync(fileUri, {UTI});
    } catch (error) {
        console.log(error);
    }

}
