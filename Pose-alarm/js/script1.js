let knnClassifier;
let video;
let canvas;
let ctx;
let poseNet;
let poses = [];
let squatState = "standing";
let squatCount = 0;
let jumpState = "onGround";
let jumpCount = 0;
let alarmTimeout;
let countdownInterval;
let requiredReps = 0;  // Add this
let selectedExercise = ""; 

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

const createButtons = () => {
  document.getElementById("addClassA").addEventListener("click", () => addData("squatting"));
  document.getElementById("addClassB").addEventListener("click", () => addData("standing"));
  document.getElementById("addClassC").addEventListener("click", () => addData("jumping"));
  document.getElementById("addClassD").addEventListener("click", () => addData("onGround"));
  document.getElementById("buttonPredict").addEventListener("click", classify);
  document.getElementById("saveModel").addEventListener("click", () => knnClassifier.save());
  document.getElementById("loadModel").addEventListener("click", loadModel);
};

const addData = (label) => {
  const poseArray = poses[0].pose.keypoints.map((p) => [p.score, p.position.x, p.position.y]);
  knnClassifier.addExample(poseArray, label);
  updateCounts();
};

const classify = () => {
  const requiredLabels = ["squatting", "standing", "jumping", "onGround"];
  const isReadyForClassification = requiredLabels.every((label) => knnClassifier.getCountByLabel()[label] > 0);

  if (isReadyForClassification) {
    const poseArray = poses[0].pose.keypoints.map((p) => [p.score, p.position.x, p.position.y]);
    knnClassifier.classify(poseArray, gotResults);
  } else {
    console.warn("Please provide samples for all classes before classifying.");
  }
};

const loadModel = () => {
  knnClassifier.load("myKNN.json", () => {
    console.log("Model loaded successfully");
    updateCounts();
    classify();
  });
};

const gotResults = (error, result) => {
  if (error) {
    console.error(error);
    return;
  }
  classifySquats(result);
  classifyJumps(result);
  updateConfidenceDisplays(result);
  classify();
};

const classifySquats = (result) => {
  if (result.label === "squatting" && result.confidencesByLabel["squatting"] >= 0.95) {
    squatState = "squatting";
  }
  if (result.label === "standing" && squatState === "squatting" && result.confidencesByLabel["standing"] >= 0.95) {
    squatState = "standing";
    squatCount++;
    document.querySelector("#squatCounter").textContent = squatCount;
  }
};

const classifyJumps = (result) => {
  if (result.label === "jumping" && result.confidencesByLabel["jumping"] >= 0.95) {
    jumpState = "jumping";
  }
  if (result.label === "onGround" && jumpState === "jumping" && result.confidencesByLabel["onGround"] >= 0.95) {
    jumpState = "onGround";
    jumpCount++;
    document.querySelector("#jumpCounter").textContent = jumpCount;
  }
};

const modelReady = () => {
  console.log("PoseNet model is ready!");
};

const resultHandler = (results) => {
  poses = results; // Store the latest poses
  drawPoses(); // Optional: Draw the poses onto the canvas
};

const drawPoses = () => {
  if (video.playing && poses.length > 0) {
    ctx.drawImage(video, 0, 0, video.width, video.height);
    let pose = poses[0].pose;
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  }
};

const updateCounts = () => {
  const counts = knnClassifier.getCountByLabel();
  // Assuming you have display elements for each class count
  document.querySelector("#squattingCount").textContent =
    counts["squatting"] || 0;
  document.querySelector("#standingCount").textContent =
    counts["standing"] || 0;
  document.querySelector("#jumpingCount").textContent = counts["jumping"] || 0;
  document.querySelector("#onGroundCount").textContent =
    counts["onGround"] || 0;
};

const updateConfidenceDisplays = (result) => {
  // Assuming you have display elements for each class confidence
  document.querySelector("#confidenceSquatting").textContent =
    (result.confidencesByLabel["squatting"] * 100).toFixed(2) + "%";
  document.querySelector("#confidenceStanding").textContent =
    (result.confidencesByLabel["standing"] * 100).toFixed(2) + "%";
  document.querySelector("#confidenceJumping").textContent =
    (result.confidencesByLabel["jumping"] * 100).toFixed(2) + "%";
  document.querySelector("#confidenceOnGround").textContent =
    (result.confidencesByLabel["onGround"] * 100).toFixed(2) + "%";
};

// Finally, call the init function to start everything:
init();
