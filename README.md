# Shape and Size Detector

A full-stack application for detecting shapes and measuring sizes in images.

## Features

- Upload images
- Detect shapes in images (triangle, square, rectangle, pentagon, circle)
- Measure object sizes in millimeters
- Beautiful React UI with three main functions

## Project Structure

```
shape-detector/
├── backend/             # Flask backend
│   ├── static/          # Static files
│   │   └── uploads/     # Uploaded and processed images
│   ├── app.py           # Main Flask application
│   ├── shapedetector.py # Shape detection module
│   └── requirements.txt # Python dependencies
│
└── frontend/            # React frontend
    ├── public/          # Public assets
    └── src/             # React source code
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:

   ```
   cd shape-detector/backend
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Run the Flask server:
   ```
   python app.py
   ```
   The server will run at http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:

   ```
   cd shape-detector/frontend
   ```

2. Install dependencies:

   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```
   The application will open at http://localhost:3000

## Usage

1. Click on the "Upload Image" button to select and upload an image
2. Click on "Detect Shape" to identify shapes in the image
3. Click on "Detect Size" to measure the dimensions of objects in the image

## Technologies Used

- **Backend**: Python, Flask, OpenCV, rembg
- **Frontend**: React, Axios, Bootstrap

## License

MIT
