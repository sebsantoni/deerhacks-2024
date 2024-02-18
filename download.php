<?php

// Set CORS headers
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST");
header("Access-Control-Allow-Headers: Content-Type");

$imgs_generated = -1;

// discard the first image, which generates weirdly for some reason
if (isset($_POST)){
    $imgs_generated++;
    if ($imgs_generated == 0){
        // pass
    }

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

    // Save the image to the specified directory
    
    if (file_put_contents($savePath, $decodedImage)) {
        echo json_encode(['success' => true, 'filename' => $filename]);
    } else {
        echo json_encode(['success' => false, 'message' => 'Error saving the image.']);
    }

 
}