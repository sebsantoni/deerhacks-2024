<?php

// Set CORS headers
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST");
header("Access-Control-Allow-Headers: Content-Type");

// POST: send image data to the server, respond with prediction
if (isset($_POST)){

    // echo "received...";

    // Get the base64-encoded image data from the FormData object
    $encodedImage = $_POST["imageData"];

    // Remove the data:image/jpeg;base64 prefix
    $encodedImage = str_replace('data:image/jpeg;base64,', '', $encodedImage);

    // Decode the base64-encoded image data
    $decodedImage = base64_decode($encodedImage);

    // Generate a unique filename
    $filename = 'image_' . uniqid() . '.jpg';

    // Specify the directory where you want to save the image
    $savePath = 'C:/Users/14379/Desktop/Coding/deerhacks-2024/user_data/' . $filename;

//     // Save the image to the specified directory
//     file_put_contents($savePath, $decodedImage);

    // echo "image saved. processing image...";

    if (file_put_contents($savePath, $decodedImage)) {
        echo json_encode(['success' => true, 'filename' => $filename]);
    } else {
        echo json_encode(['success' => false, 'message' => 'Error saving the image.']);
    }

    # make a prediction for the image
    $command = escapeshellcmd("python C:/Users/14379/Desktop/Coding/deerhacks-2024/predictor.py $savePath");
    exec($command, $output);
    print_r($output);

    
}

// // GET: Send prediction to server, get ecological score
// if (isset($_GET)){

// }

   