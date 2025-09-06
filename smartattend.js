// ====== Backend (Node.js + Express) ======
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');

const faceapi = require('face-api.js');
const canvas = require('canvas');
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const PORT = 3001;

// Serve static frontend HTML below
app.use(express.static(__dirname + '/public'));

app.use(bodyParser.json({limit: '8mb'}));
app.use(cors());

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Student schema/model
const StudentSchema = new mongoose.Schema({
  id: { type: String, required: true, unique: true },
  name: String,
  email: String,
  class: String,
  enrolled: Boolean,
  photoData: String,
  faceFeatures: { type: [Number] },
  enrollmentDate: String,
});
const Student = mongoose.model('Student', StudentSchema);

// Load face-api.js models
const MODEL_PATH = path.join(__dirname, '/models');
(async () => {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
  console.log('Face-api.js models loaded!');
})();

// API endpoints

app.post('/api/student/enroll', async (req, res) => {
  const { id, name, email, class: className, photoData } = req.body;
  try {
    const img = await canvas.loadImage(photoData);
    const result = await faceapi.detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (!result) return res.status(400).json({ success: false, message: "Face not detected." });

    const newStudent = new Student({
      id,
      name,
      email,
      class: className,
      enrolled: true,
      photoData,
      faceFeatures: Array.from(result.descriptor),
      enrollmentDate: new Date().toISOString().split('T')[0],
    });

    await newStudent.save();
    res.json({ success: true, message: "Student enrolled." });
  } catch (e) {
    res.status(400).json({ success: false, message: "Error enrolling student: " + e.message });
  }
});

app.get('/api/student/all', async (req, res) => {
  const students = await Student.find({});
  res.json(students);
});

app.post('/api/student/attendance', async (req, res) => {
  const { imageData } = req.body;
  try {
    const img = await canvas.loadImage(imageData);
    const queryResult = await faceapi.detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (!queryResult) return res.status(400).json({ success: false, message: "Face not detected." });

    const students = await Student.find({ faceFeatures: { $exists: true, $ne: null } });
    let bestMatch = null;
    let minDistance = 0.6;
    students.forEach(stu => {
      const dist = faceapi.euclideanDistance(queryResult.descriptor, stu.faceFeatures);
      if (dist < minDistance) {
        minDistance = dist;
        bestMatch = stu;
      }
    });

    if (bestMatch) {
      return res.json({
        success: true,
        studentName: bestMatch.name,
        studentId: bestMatch.id,
        confidence: (100 - minDistance * 100).toFixed(2),
        message: `Attendance marked: ${bestMatch.name} (Confidence: ${(100 - minDistance * 100).toFixed(2)}%)`,
      });
    } else {
      return res.status(404).json({ success: false, message: "No matching student found." });
    }
  } catch (e) {
    res.status(400).json({ success: false, message: "Attendance error: " + e.message });
  }
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/public/index.html'));
});

app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));

// ====== Frontend (public/index.html) ======
/*
Create a folder named 'public' next to smartattend.js.
Save the following HTML as public/index.html
Open http://localhost:3001 in your browser.
*/

