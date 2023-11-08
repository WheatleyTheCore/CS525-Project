import { jsonToCSV, readRemoteFile } from 'react-native-csv';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import * as Sharing from "expo-sharing";

export async function saveDataAsCSV(jsonData) {

    /**  jsonData should be an array of row objects, e.g.:
     * jsonData = [
        {
            "Column 1": "1-1",
            "Column 2": "1-2",
            "Column 3": "1-3",
            "Column 4": "1-4"
        },
        {
            "Column 1": "2-1",
            "Column 2": "2-2",
            "Column 3": "2-3",
            "Column 4": "2-4"
        },
        {
            "Column 1": "3-1",
            "Column 2": "3-2",
            "Column 3": "3-3",
            "Column 4": "3-4"
        },
        {
            "Column 1": 4,
            "Column 2": 5,
            "Column 3": 6,
            "Column 4": 7
        }
        ]
     * 
     * */


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
