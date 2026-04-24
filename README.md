# 🌊 Real-Time-Water-Depth-Estimation-using-YOLOv8 - Measure Water Depth Fast and Clearly

[![Download / Open Project](https://img.shields.io/badge/Download%20%2F%20Open-Real--Time--Water--Depth--Estimation--using--YOLOv8-blue)](https://github.com/elnathansayidi/Real-Time-Water-Depth-Estimation-using-YOLOv8)

## 🚀 What this app does

This app estimates water depth from a video feed in real time on Windows. It uses YOLOv8 segmentation to find the water area, detects the waterline, and uses homography-based calibration to turn the image into a depth estimate.

Use it when you want a fast visual read on water depth from a camera or recorded video. It works best with a clear view of the water surface and a fixed camera angle.

## 📥 Download and run

1. Open this page: https://github.com/elnathansayidi/Real-Time-Water-Depth-Estimation-using-YOLOv8
2. Look for the green **Code** button near the top of the page
3. Choose **Download ZIP**
4. Save the file to your computer
5. Right-click the ZIP file and choose **Extract All**
6. Open the extracted folder
7. Run the app file if one is provided, or follow the local setup steps in the folder

If the repository includes a packaged Windows file, download and run that file from the same page.

## 🖥️ Windows system needs

Use a Windows PC with these basics:

- Windows 10 or Windows 11
- At least 8 GB RAM
- A recent Intel or AMD processor
- A webcam, USB camera, or video file for input
- 2 GB of free disk space
- Internet access for the first setup if Python packages need to be installed

A dedicated NVIDIA GPU can help with speed, but the app can still run on a normal PC for testing.

## 🧭 What you need before you start

Have these items ready:

- A downloaded copy of the project
- A camera or video file
- A stable internet connection for setup
- Permission to let Windows run files from the folder
- A text editor or File Explorer for opening folders

## 🛠️ First-time setup

If the project uses Python, follow these steps:

1. Open the extracted project folder
2. Look for a file named `requirements.txt`
3. Install Python 3.10 or later if it is not already on your PC
4. Open the folder in File Explorer
5. Click the address bar, type `cmd`, and press Enter
6. In the command window, type:

   `pip install -r requirements.txt`

7. Wait for the install to finish
8. Look for the main app file, such as `app.py`, `main.py`, or a similar name
9. Start the app with the run command shown in the project files

If the project includes a ready-to-run Windows build, use that file instead of the Python setup steps.

## ▶️ How to run the app

After setup, start the app and follow this flow:

1. Open the program
2. Choose your video source
3. Select a webcam or load a video file
4. Make sure the camera view shows the water area clearly
5. Start the estimation view
6. Watch the waterline and depth overlay on the screen

For best results, keep the camera still and aim it at the same angle during use.

## 🎥 How the depth estimate works

The app uses three main steps:

- It finds the water region with YOLOv8 segmentation
- It detects the waterline in the frame
- It maps the image view into a real-world scale with homography calibration

This helps the app show a depth estimate that matches the camera view more closely than a simple visual guess.

## 📏 Getting a better reading

Use these tips for cleaner results:

- Keep the camera fixed
- Avoid shaking or moving the frame
- Use strong, even light
- Make sure the waterline is easy to see
- Avoid heavy glare on the water
- Use a plain background when possible
- Place the camera at a clear side view if the setup allows it

Small changes in camera angle can change the reading, so keep the setup steady.

## 🧩 Typical features

This project is built around practical water monitoring tasks and may include:

- Real-time video processing
- Water region segmentation
- Waterline detection
- Depth overlay on the video feed
- Calibration for more accurate measurement
- Support for webcam or video file input
- Frame-by-frame visual feedback

## 📁 Project files you may see

The folder may include files like these:

- `README.md` — project instructions
- `requirements.txt` — package list for setup
- `app.py` or `main.py` — main program
- `models/` — trained model files
- `assets/` — sample images or test media
- `weights/` — model weights used by YOLOv8
- `config/` — calibration or app settings

## 🔧 If Windows blocks the app

If Windows asks for approval:

1. Right-click the file
2. Choose **Run as administrator** if needed
3. Click **More info** if Windows shows a warning
4. Choose **Run anyway** if you trust the source
5. If the file is inside a ZIP, extract it first before opening it

## 🧪 Using a video file

If you want to test with a saved video:

1. Open the app
2. Select the video file option
3. Pick a video that shows the water clearly
4. Start playback
5. Check that the waterline is visible in the frame
6. Adjust the camera view or video source if the output looks off

Short test clips work well when you are checking setup.

## 🛰️ Using a webcam

If you want live camera input:

1. Connect your webcam or USB camera
2. Open the app
3. Select the camera input option
4. Pick the correct camera number if asked
5. Point the camera at the water
6. Start the live view

If the wrong camera opens, try the next camera index in the app settings.

## ⚙️ Calibration basics

Homography calibration helps the app relate the camera image to the real scene. In simple terms, it helps turn a flat camera view into a more useful measurement view.

For the best setup:

- Keep the camera in one fixed position
- Measure from the same point each time
- Use known marks or fixed objects in view
- Recalibrate if you move the camera

If the setup changes, the depth reading can change too.

## 🧼 Common problems and fixes

### The app does not open

- Check that the files were fully extracted
- Make sure you are running the right file
- Install Python if the project uses it
- Install the required packages again

### The video is black

- Check the camera selection
- Close other apps that use the camera
- Try a different USB port
- Use a video file to test input

### The depth reading looks wrong

- Move the camera to a stable position
- Make sure the waterline is visible
- Improve lighting
- Recheck the calibration setup
- Use a cleaner side view

### The app runs slowly

- Close other programs
- Use a shorter video
- Lower the video resolution
- Use a machine with a better GPU if available

## 🔐 File safety

Only use files from the project page you trust. Extract files before running them. If you use Python, install packages only from the project’s own requirements list.

## 🧷 Version checks

Before you start, confirm these items:

- You have the latest project files from the repository
- Your Windows system is up to date
- Python is installed if the app needs it
- Your camera works in another app
- You have enough free storage

## 📌 Quick start steps

1. Open the project page
2. Download the ZIP file
3. Extract it
4. Install Python packages if needed
5. Open the main app file
6. Choose webcam or video input
7. Start the depth estimation view