let knnClassifier;
let inputs;
let video;
let canvas;
let ctx;
let poseNet;
let currentExercise = ""; // Track the current exercise (either "squats" or "")

const keypointNames = [
  "leftShoulder",
  "leftElbow",
  "leftWrist",
  "rightShoulder",
  "rightElbow",
  "rightWrist",
  "nose",
  "leftEye",
  "rightEye",
];

const init = async () => {
  knnClassifier = ml5.KNNClassifier();

  video = document.querySelector(".video");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });
  video.srcObject = stream;
  video.play();

  canvas = document.querySelector(".canvas");
  canvas.width = video.width;
  canvas.height = video.height;
  ctx = canvas.getContext("2d");

  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on("pose", resultHandler);

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
  const exerciseLabel = result.label;

  if (exerciseLabel === "squats") {
    if (currentExercise !== "squats") {
      currentExercise = "squats";
      console.log("Squats started.");
    } else {
      console.log("Squat counted.");
    }
  }

  document.querySelector(
    ".status"
  ).textContent = `Current Exercise: ${currentExercise}`;
};

const addDataHandler = () => {
  const $label = document.querySelector(".label");
  const output = $label.value;

  if (output === "pushups" || output === "squats") {
    knnClassifier.addExample(inputs, output);
  } else {
    console.log('Invalid exercise label. Please enter "pushups" or "squats".');
  }
};

const resultHandler = (poses) => {
  if (!poses.length) return;

  const pose = poses[0].pose;
  const keypoints = pose.keypoints;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "red";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;

  for (const keypoint of keypoints) {
    const keyPointNeedsTobeRendered = keypointNames.includes(keypoint.part);
    if (keyPointNeedsTobeRendered) {
      const { x, y } = keypoint.position;
      if (keypoint.part === "nose") {
        // console.log(x, y);
      }
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }

  inputs = keypoints
    .filter((kp) => keypointNames.includes(kp.part))
    .map((keypoint) => [keypoint.position.x, keypoint.position.y]);
};

const modelReady = () => {
  console.log("PoseNet model loaded");
  poseNet.multiPose(video);
};

init();
