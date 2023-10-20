let knnClassifier;
let video;
let canvas;
let ctx;
let poseNet;
let poses = [];
let squatState;
let squatCount = 0;
let alarmTimeout;
let countdownInterval;
let requiredReps = 0;
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
  if (
    knnClassifier.getNumLabels() > 0 &&
    knnClassifier.getCountByLabel()["A"] > 0 &&
    knnClassifier.getCountByLabel()["B"] > 0
  ) {
    const poseArray = poses[0].pose.keypoints.map((p) => [
      p.score,
      p.position.x,
      p.position.y,
    ]);
    knnClassifier.classify(poseArray, gotResults);
  } else {
    console.warn("Please provide samples for both classes before classifying.");
  }
};

const createButtons = () => {
  document
    .getElementById("addClassA")
    .addEventListener("click", () => addData("A"));
  document
    .getElementById("addClassB")
    .addEventListener("click", () => addData("B"));
  document.getElementById("buttonPredict").addEventListener("click", () => {
    if (
      knnClassifier.getCountByLabel()["A"] &&
      knnClassifier.getCountByLabel()["B"]
    ) {
      classify();
    } else {
      console.warn(
        "Please add examples for both classes before starting prediction."
      );
    }
  });
  // document.getElementById("clearAll").addEventListener("click", clearAllLabels);
  document
    .getElementById("saveModel")
    .addEventListener("click", () => knnClassifier.save());
  document.getElementById("loadModel").addEventListener("click", () => {
    knnClassifier.load("myKNN.json", () => {
      console.log("Model loaded successfully");
      updateCounts();
      classify();
    });
  });
};

const gotResults = (error, result) => {
  if (error) {
    console.error(error);
    return;
  }

  if (
    result.label === "B" &&
    (squatState === "standing" || squatState === undefined) &&
    result.confidencesByLabel["B"] >= 0.95
  ) {
    squatState = "squatting";
  }

  else if (
    result.label === "A" &&
    squatState === "squatting" &&
    result.confidencesByLabel["A"] >= 0.95
  ) {
    squatState = "standing";
    squatCount++;
    document.querySelector("#squatCounter").textContent = squatCount;
  }

  else if (squatState === undefined && result.confidencesByLabel["A"] >= 0.95) {
    squatState = "standing"; // Default starting state if classifier is confident the user is standing
  } else if (
    squatState === undefined &&
    result.confidencesByLabel["B"] >= 0.95
  ) {
    squatState = "squatting"; // Default starting state if classifier is confident the user is squatting
  }

  checkExerciseCompletion();
  updateConfidenceDisplays(result);
  classify();
};

const updateCounts = () => {
  const counts = knnClassifier.getCountByLabel();
  document.getElementById("exampleA").textContent = counts["A"] || 0;
  document.getElementById("exampleB").textContent = counts["B"] || 0;
  // document.getElementById("exampleC").textContent = counts["C"] || 0;
  // document.getElementById("exampleD").textContent = counts["D"] || 0;
};

const clearAllLabels = () => {
  knnClassifier.clearAllLabels();
  updateCounts();
};

const resultHandler = (results) => {
  if (!results.length) return;
  poses = results;
  drawKeypoints();
};

const modelReady = () => {
  console.log("PoseNet model loaded");
  poseNet.multiPose(video);
};

const drawKeypoints = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "red";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  const keypoints = poses[0].pose.keypoints;

  keypoints.forEach((keypoint) => {
    ctx.beginPath();
    ctx.arc(keypoint.position.x, keypoint.position.y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  });
};

const checkExerciseCompletion = () => {
  if (selectedExercise === "squat" && squatCount >= requiredReps) {
    document.getElementById("alarmSound").pause();
    document.getElementById("alarmSound").currentTime = 0; // reset the sound time
    requiredReps = 0;
    squatCount = 0;
    document.querySelector("#squatCounter").textContent = 0;
  } // Similarly, add logic for jumps when you handle that
};

const updateConfidenceDisplays = (result) => {
  document.querySelector("#result").textContent = result.label;
  document.querySelector("#confidence").textContent = `${(
    result.confidencesByLabel[result.label] * 100
  ).toFixed(2)}%`;
  document.getElementById("confidenceA").textContent = `${(
    result.confidencesByLabel["A"] * 100
  ).toFixed(2)}%`;
  document.getElementById("confidenceB").textContent = `${(
    result.confidencesByLabel["B"] * 100
  ).toFixed(2)}%`;
};

init();

const setAlarm = () => {
  const alarmInput = document.getElementById("alarmTime");
  const alarmTime = new Date();
  const [hour, minute] = alarmInput.value.split(":").map(Number);
  const exerciseChoice = document.getElementById("exerciseChoice").value;
  const repetitionCount = document.getElementById("repetitionCount").value;

  selectedExercise = exerciseChoice;
  requiredReps = parseInt(repetitionCount);
  alarmTime.setHours(hour);
  alarmTime.setMinutes(minute);
  alarmTime.setSeconds(0);
  alarmTime.setMilliseconds(0);

  const currentTime = new Date();
  if (alarmTime <= currentTime) {
    alarmTime.setDate(alarmTime.getDate() + 1); // set for next day if time already passed
  }

  const displayTimeUntilAlarm = () => {
    const currentTime = new Date();
    const difference = alarmTime - currentTime;
    if (difference <= 0) {
      document.querySelector(".current-time h2").innerText = "ALARM!";
      clearInterval(countdownInterval);
      return;
    }
    const hours = Math.floor(difference / 1000 / 60 / 60);
    const minutes = Math.floor((difference / 1000 / 60) % 60);
    const seconds = Math.floor((difference / 1000) % 60);

    const timeString = `${String(hours).padStart(2, "0")}:${String(
      minutes
    ).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

    document.querySelector(".current-time h2").innerText = timeString;
  };

  if (countdownInterval) {
    clearInterval(countdownInterval);
  }

  countdownInterval = setInterval(displayTimeUntilAlarm, 1000);

  const feedback = document.getElementById("feedback");
  feedback.style.display = "block";
  setTimeout(() => {
    feedback.style.display = "none";
  }, 3000);

  const durationUntilAlarm = alarmTime - currentTime;
  console.log(durationUntilAlarm);

  if (alarmTimeout) {
    clearTimeout(alarmTimeout);
  }

  alarmTimeout = setTimeout(() => {
    const alarmSound = document.getElementById("alarmSound");
    alarmSound.play();
  }, durationUntilAlarm);
};

document.getElementById("alarmTime").addEventListener("change", setAlarm);
