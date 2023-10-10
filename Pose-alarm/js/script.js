let knnClassifier;
let inputs;

const init = async () => {
  knnClassifier = ml5.KNNClassifier();

  const $video = document.querySelector(".video");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });
  $video.srcObject = stream;
  $video.play();

  // Initialize PoseNet
  const poseNet = ml5.poseNet($video, modelReady);
  poseNet.on("pose", resultHandler); // Listen for pose detection

  const $addDataButton = document.querySelector(".addData");
  $addDataButton.addEventListener("click", addDataHandler);

  const $classifyButton = document.querySelector(".classify");
  $classifyButton.addEventListener("click", classifyHandler);
};

const classifyHandler = () => {
  knnClassifier.classify(inputs, classifyResultHandler);
};

const classifyResultHandler = (error, result) => {
  if (error) return console.error(error);
  console.log(result.label);
  document.querySelector(".status").textContent = result.label;

  // Check if the exercise label matches the expected exercise
  if (result.label === "pushups" || result.label === "squats") {
    // Execute alarm turn-off logic here (e.g., stop alarm sound)
    // Implement exercise validation and count here
    // If the required exercise count is reached, turn off the alarm.
  }
};

const addDataHandler = () => {
  const $label = document.querySelector(".label");
  const output = $label.value;

  // Check if the exercise label is valid ('pushups' or 'squats')
  if (output === "pushups" || output === "squats") {
    knnClassifier.addExample(inputs, output);
  } else {
    console.log('Invalid exercise label. Please enter "pushups" or "squats".');
  }
};

const resultHandler = (poses) => {
  if (!poses.length) return;

  // Use the first detected pose for input data
  const pose = poses[0].pose;
  const keypoints = pose.keypoints.map((keypoint) => [
    keypoint.position.x,
    keypoint.position.y,
  ]);
  inputs = keypoints.flat();
};

const modelReady = () => {
  console.log("PoseNet model loaded");
};

init();
