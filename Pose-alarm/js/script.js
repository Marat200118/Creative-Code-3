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
  document.querySelector("#alarmTime").addEventListener("change", setAlarm);
  document
    .querySelector("#addClassA")
    .addEventListener("click", () => addData("A"));
  document
    .querySelector("#addClassB")
    .addEventListener("click", () => addData("B"));
  document.querySelector("#buttonPredict").addEventListener("click", () => {
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

  document
    .querySelector("#saveModel")
    .addEventListener("click", () => knnClassifier.save());
  document.querySelector("#loadModel").addEventListener("click", () => {
    knnClassifier.load("myKNN.json", () => {
      console.log("Model loaded successfully");
      updateCounts();
      setTimeout(classify, 1000);
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
  } else if (
    result.label === "A" &&
    squatState === "squatting" &&
    result.confidencesByLabel["A"] >= 0.95
  ) {
    squatState = "standing";
    squatCount++;
    updateSquatDisplay();
  } else if (
    squatState === undefined &&
    result.confidencesByLabel["A"] >= 0.95
  ) {
    squatState = "standing";
  } else if (
    squatState === undefined &&
    result.confidencesByLabel["B"] >= 0.95
  ) {
    squatState = "squatting";
  }

  checkExerciseCompletion();
  updateConfidenceDisplays(result);
  classify();
};

const updateSquatDisplay = () => {
  const squatCounter = document.querySelector("#squatCounter");
  squatCounter.textContent = `${squatCount}/${requiredReps}`;
  if (squatCount < requiredReps) {
    squatCounter.style.color = "red";
  } else {
    squatCounter.style.color = "green";
  }
};

const updateCounts = () => {
  const counts = knnClassifier.getCountByLabel();
  document.querySelector("#exampleA").textContent = counts["A"] || 0;
  document.querySelector("#exampleB").textContent = counts["B"] || 0;
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
    document.querySelector("#alarmSound").pause();
    document.querySelector("#alarmSound").currentTime = 0;
    document.querySelector(".current-time h2").innerText = "ALARM TURNED OFF!";
    document.querySelector(".current-time h2").style.color = "green";
    requiredReps = 0;
    squatCount = 0;
    updateSquatDisplay();
    document.querySelector("#squatCounter").textContent = 0;
  }
};

const updateConfidenceDisplays = (result) => {
  document.querySelector(".confidenceA").textContent = `${(
    result.confidencesByLabel["A"] * 100
  ).toFixed(2)}%`;
  document.querySelector(".confidenceB").textContent = `${(
    result.confidencesByLabel["B"] * 100
  ).toFixed(2)}%`;
};

const setAlarm = () => {
  const alarmInput = document.querySelector("#alarmTime");
  const alarmTime = new Date();
  const [hour, minute] = alarmInput.value.split(":").map(Number);
  const exerciseChoice = document.querySelector("#exerciseChoice").value;
  const repetitionCount = document.querySelector("#repetitionCount").value;

  selectedExercise = exerciseChoice;
  requiredReps = parseInt(repetitionCount);
  alarmTime.setHours(hour);
  alarmTime.setMinutes(minute);
  alarmTime.setSeconds(0);
  alarmTime.setMilliseconds(0);

  const currentTime = new Date();
  if (alarmTime <= currentTime) {
    alarmTime.setDate(alarmTime.getDate() + 1);
  }

  const displayTimeUntilAlarm = () => {
    const currentTime = new Date();
    const difference = alarmTime - currentTime;
    if (difference <= 0) {
      document.querySelector(".current-time h2").innerText = "ALARM!";
      document.querySelector(".current-time h2").style.color = "red";
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
  const feedback = document.querySelector("#feedback");
  feedback.style.display = "block";
  setTimeout(() => {
    feedback.style.display = "none";
  }, 3000);

  const durationUntilAlarm = alarmTime - currentTime;

  if (alarmTimeout) {
    clearTimeout(alarmTimeout);
  }

  alarmTimeout = setTimeout(() => {
    const alarmSound = document.querySelector("#alarmSound");
    alarmSound.play();
  }, durationUntilAlarm);
  updateSquatDisplay();
};

init();
