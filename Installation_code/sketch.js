let video;
let poseNet;
let poses = [];

let faceapi;
let detections = [];

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  const faceOptions = {
    withLandmarks: true,
    withExpressions: true,
    withDescriptors: true,
    minConfidence: 0.5,
  };

  faceapi = ml5.faceApi(video, faceOptions, faceReady);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on("pose", function(results) {
    poses = results;
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  select("#status").html("Model Loaded");
}

function faceReady() {
  faceapi.detect(gotFaces);
}

function gotFaces(error, result) {
  if (error) {
    faceapi.detect(gotFaces);
    
  }

  detections = result;
  clear();
  drawExpressions(detections);
  faceapi.detect(gotFaces);
}

function drawExpressions(detections) {
  let { neutral, happy, angry, sad, disgusted, surprised, fearful } = detections[0].expressions;
  let emotion = [neutral, happy, angry, sad, disgusted, surprised, fearful];

  let color = [
    [255, 255, 255],
    [255, 255, 0],
    [255, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 255],
    [0, 255, 255]
  ];

  for (let i = 0; i < emotion.length; i++) {
    if (emotion[i] > 0.6) {
      draw(color[i]);
    }
  }
}

function draw(color) {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints and the skeletons
  drawKeypoints(color);
  drawSkeleton(color);
}


// A function to draw ellipses over the detected keypoints
function drawKeypoints(color) {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i += 1) {
    // For each pose detected, loop through all the keypoints
    const pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j += 1) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      const keypoint = pose.keypoints[j];
      let eyeR = pose.rightEye;
      let eyeL = pose.leftEye;
      let d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y);
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(color);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, d);
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton(color) {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i += 1) {
    const skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j += 1) {
      const partA = skeleton[j][0];
      const partB = skeleton[j][1];
      const eyeR = poses[i].pose.rightEye;
      const eyeL = poses[i].pose.leftEye;
      const d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y);
      const strokeWeightValue = map(d, 0, width, 1, 10); // Map the distance to stroke weight range

      stroke(color);
      strokeWeight(strokeWeightValue * 80);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}