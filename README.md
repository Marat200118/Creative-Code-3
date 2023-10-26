# Pose Exercise Alarm 🏋️‍♂️⏰
Devine - Creative code 3 course

A unique alarm application that utilizes machine learning models to track user movements and turn off the alarm only when the user completes a set number of exercises!


# Overview
The Pose Exercise Alarm system combines the power of the KNN classifier with the PoseNet model to detect user-defined movements. Users can set an alarm and the only way to turn it off is by doing the specified number of squats.

# Features
1. Live Video Feed: Capture and display the user's pose in real-time.
2. Customizable Alarm Settings: Set the alarm time, choose an exercise, and define the number of repetitions.
3. Pose Detection: Utilizes the PoseNet model for accurate pose detection.
4. Interactive Interface: Visual feedback on the number of repetitions and the confidence level of the pose detection.
5. Model Training & Saving: Ability to add data for poses and save/load the KNN model.

# Usage
1. Set the Alarm: Select the desired alarm time.
2. Choose an Exercise: For now, we support squats but more exercises can be added in the future.
3. Define Repetitions: Set the number of repetitions required to turn off the alarm.
4. Train the Model:
-- Click the "Class A" button while performing the exercise (e.g., in a squat position).
-- Click the "Class B" button while standing up.
-- Add enough samples for both classes for accurate predictions.
5. Save/Load Model: After training, you can save the model for future use and load it whenever needed.

When the alarm time arrives, perform the specified number of repetitions to stop the alarm.