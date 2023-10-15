let knnClassifier;
let video;
let canvas;
let ctx;
let poseNet;
let poses = [];

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
  "leftHip",
  "rightHip",
  "leftKnee",
  "rightKnee",
  "leftAnkle",
  "rightAnkle",
];

const init = async () => {
  knnClassifier = ml5.KNNClassifier();
  video = document.querySelector("#video");
  canvas = document.querySelector("#canvas");
  ctx = canvas.getContext("2d");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });

  video.srcObject = stream;
  video.play();

  video.addEventListener("loadedmetadata", function () {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  });

  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on("pose", resultHandler);
  createButtons();
  updateCounts();
};

const addData = (label) => {
  const poseArray = poses[0].pose.keypoints.map((p) => [
    p.score,
    p.position.x,
    p.position.y,
  ]);
  knnClassifier.addExample(poseArray, label);
  updateCounts();
};

const classify = () => {
  if (knnClassifier.getNumLabels() > 0) {
    const poseArray = poses[0].pose.keypoints.map((p) => [
      p.score,
      p.position.x,
      p.position.y,
    ]);
    knnClassifier.classify(poseArray, gotResults);
  }
};

const createButtons = () => {
  document
    .getElementById("addClassA")
    .addEventListener("click", () => addData("A"));
  document
    .getElementById("addClassB")
    .addEventListener("click", () => addData("B"));
  document
    .getElementById("resetA")
    .addEventListener("click", () => clearLabel("A"));
  document
    .getElementById("resetB")
    .addEventListener("click", () => clearLabel("B"));
  document.getElementById("buttonPredict").addEventListener("click", classify);
  document.getElementById("clearAll").addEventListener("click", clearAllLabels);
};

const gotResults = (error, result) => {
  if (error) {
    console.error(error);
    return;
  }
  const confidences = result.confidencesByLabel;
  document.querySelector("#result").textContent = result.label;
  document.querySelector("#confidence").textContent = `${(
    confidences[result.label] * 100
  ).toFixed(2)}%`;
  document.getElementById("confidenceA").textContent = `${(
    confidences["A"] * 100
  ).toFixed(2)}%`;
  document.getElementById("confidenceB").textContent = `${(
    confidences["B"] * 100
  ).toFixed(2)}%`;
  classify(); // keep classifying
};

const updateCounts = () => {
  const counts = knnClassifier.getCountByLabel();
  document.getElementById("exampleA").textContent = counts["A"] || 0;
  document.getElementById("exampleB").textContent = counts["B"] || 0;
};

const clearLabel = (classLabel) => {
  knnClassifier.clearLabel(classLabel);
  updateCounts();
};
 
const clearAllLabels = () => {
  knnClassifier.clearAllLabels();
  updateCounts();
};

const resultHandler = (results) => {
  if (!results.length) return;
  poses = results;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "red";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  const keypoints = results[0].pose.keypoints;

  keypoints.forEach((keypoint) => {
    if (keypointNames.includes(keypoint.part)) {
      ctx.beginPath();
      ctx.arc(keypoint.position.x, keypoint.position.y, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  });
};

const modelReady = () => {
  console.log("PoseNet model loaded");
  poseNet.multiPose(video);
};

init();
