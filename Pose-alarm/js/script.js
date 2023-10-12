let knnClassifier;
let video;
let canvas;
let ctx;
let poseNet;
let squatCount = 0;
let lastPose = 0;
const exerciseLabel = "squat down";

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

let pose;

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
  video.addEventListener("loadedmetadata", function () {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  });
  ctx = canvas.getContext("2d");

  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on("pose", resultHandler);

  updateCountDisplay();

  const $addDataButton = document.querySelector(".addData");
  $addDataButton.addEventListener("click", addSquatDataHandler);

  const $resetSquatCountButton = document.querySelector(".resetSquatCount");
  $resetSquatCountButton.addEventListener("click", resetSquatCount);

  const $classifySquatButton = document.querySelector(".classifySquat");
  $classifySquatButton.addEventListener("click", classifySquatHandler);
};

const updateCountDisplay = () => {
  document.querySelector(".status").textContent = `Squats: ${squatCount}`;
};

const classifySquatHandler = () => {
  classifySquatPosition();
};

const classifySquatPosition = () => {
    // if (pose) {
    //   const numExamples = knnClassifier.getNumExamples();
    //   if (numExamples <= 0) {
    //     console.log("No examples in any label");
    //     return;
    //   }
    // }
  
  if (pose) {
    const inputs = keypointNames.map((part) =>
      pose[part] ? [pose[part].x, pose[part].y] : [0, 0]
    );

    knnClassifier.classify(inputs, (error, result) => {
      if (error) {
        console.error(error);
        return;
      }

      // console.log("Classification result:", result);

      const squatLabel = result.label;

      if (squatLabel === exerciseLabel) {
        if (lastPose === 1) {
          squatCount++;
          lastPose = 0;
          updateCountDisplay();
        }
      } else {
        lastPose = 1;
      }
    });
  }
};

const addSquatDataHandler = () => {
  console.log("Adding data for label:", exerciseLabel);
  if (!pose) {
    console.log("No pose detected.");
    return;
  }
  if (pose) {
    const exerciseLabelInput = document.querySelector(".label");
    const inputLabel = exerciseLabelInput.value;
    const inputs = keypointNames.map((part) =>
      pose[part] ? [pose[part].x, pose[part].y] : [0, 0]
    );

    console.log(
      `Added data for label: ${exerciseLabel}, with keypoints:`,
      inputs
    );

    knnClassifier.addExample(inputs, exerciseLabel);

    const dataset = knnClassifier.getClassifierDataset();
    if (dataset) {
      Object.keys(dataset).forEach((key) => {
        console.log(`Label: ${key}`);
        const data = dataset[key].dataSync();
        console.log("Data:", Array.from(data));
      });
    }

    const notification = document.querySelector(".notification");
    notification.textContent = `Data added for ${exerciseLabel}!`;
    setTimeout(() => {
      notification.textContent = ""; // Clear the message after 3 seconds
    }, 3000);
  }
};

const resetSquatCount = () => {
  squatCount = 0;
  updateCountDisplay();
};

const resultHandler = (poses) => {
  if (!poses.length) return;

  pose = poses[0].pose;
  const keypoints = pose.keypoints;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "red";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;

  for (const keypoint of keypoints) {
    const keyPointNeedsToBeRendered = keypointNames.includes(keypoint.part);
    if (keyPointNeedsToBeRendered) {
      const { x, y } = keypoint.position;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }
};



const modelReady = () => {
  console.log("PoseNet model loaded");
  poseNet.multiPose(video);
};

init();
